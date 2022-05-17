# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Code was heavily based on https://github.com/microsoft/CvT

import logging
from functools import partial

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import TruncatedNormal, Constant, XavierUniform

from ppcls.utils.save_load import load_dygraph_pretrain, load_dygraph_pretrain_from_url


MODEL_URLS = {
    "CvT_13_224":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/CvT_13_224_pretrained.pdparams",
    "CvT_13_384":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/CvT_13_384_pretrained.pdparams",
    "CvT_21_224":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/CvT_21_224_pretrained.pdparams",
    "CvT_21_384":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/CvT_21_384_pretrained.pdparams",
    "CvT_w24_384":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/CvT_w24_384_pretrained.pdparams",
}

__all__ = list(MODEL_URLS.keys())


zeros_ = Constant(value=0.)
ones_ = Constant(value=1.)
xavier_uniform_ = XavierUniform()
trunc_normal_ = TruncatedNormal(std=.02)


def rearrange(tensor, pattern: str, **axes_lengths):
    if pattern == 'b l (h d) -> b h l d' and tensor.ndim == 3:
        h = axes_lengths['h']
        b, l, hd = tensor.shape
        d = hd // h
        assert h * d == hd
        tensor = tensor.reshape((b, l, h, d)).transpose([0, 2, 1, 3])
    elif pattern == 'b h l d -> b l (h d)' and tensor.ndim == 4:
        tensor = tensor.transpose([0, 2, 1, 3]).flatten(2)
    elif pattern == 'b (h w) c -> b c h w' and tensor.ndim == 3:
        h = axes_lengths['h']
        w = axes_lengths['w']
        b, hw, c = tensor.shape
        assert hw == h * w
        tensor = tensor.transpose([0, 2, 1]).reshape([b, c, h, w])
    elif pattern == 'b c h w -> b (h w) c':
        tensor = tensor.flatten(2).transpose([0, 2, 1])
    return tensor


def drop_path(x, drop_prob=0., training=False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ...
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = paddle.to_tensor(1 - drop_prob)
    shape = (paddle.shape(x)[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = keep_prob + paddle.rand(shape, dtype=x.dtype)
    random_tensor = paddle.floor(random_tensor)  # binarize
    output = x.divide(keep_prob) * random_tensor
    return output


class DropPath(nn.Layer):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Rearrange(nn.Layer):
    def __init__(self, pattern):
        super(Rearrange, self).__init__()
        assert pattern == 'b c h w -> b (h w) c'

    def forward(self, x):
        # rearrange x 'b c h w -> b (h w) c'
        x = x.flatten(2).transpose([0, 2, 1])
        return x


class RearrangeNew(nn.Layer):
    def __init__(self, pattern, **axes_lengths):
        super(Rearrange, self).__init__()
        self.pattern = pattern
        self.axes_lengths = axes_lengths

    def forward(self, x):
        # rearrange x 'b c h w -> b (h w) c'
        x = rearrange(x, self.pattern, **self.axes_lengths)
        return x


class DepthwiseSeparableConv2D(nn.Layer):
    def __init__(self, dim_in, dim_out, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise_conv=nn.Conv2D(
            in_channels=dim_in,
            out_channels=dim_in,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=dim_in
        )
        self.pointwise_conv=nn.Conv2D(
            in_channels=dim_in,
            out_channels=dim_out,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )
        
    def forward(self, x):
        out=self.depthwise_conv(x)
        out=self.pointwise_conv(out)
        return out


class LayerNorm(nn.LayerNorm):
    """Subclass paddle's LayerNorm to handle fp16."""
    def __init__(self, normalized_shape, epsilon=0.00001, weight_attr=None, bias_attr=None, name=None):
        super().__init__(normalized_shape, epsilon, weight_attr, bias_attr, name)
    
    def forward(self, x):
        orig_type = x.dtype
        ret = super().forward(x.astype(paddle.float32))
        return ret.astype(orig_type)
    

class QuickGELU(nn.Layer):
    def forward(self, x):
        return x * F.sigmoid(1.702 * x)


class Mlp(nn.Layer):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Layer):
    r""" Convolutions based multi-head self attention (MSA) module.

    Args:
        dim_in (int): Number of input channels.
        dim_out (int): Number of output channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        method (str, optional): init method. Default: dw_bn
        kernel_size (tuple[int]): Number of kernel size.
        stride_kv (int): Number of kv stride. Default: 1.
        stride_q (int): Number of q stride. Default: 1.
        padding_kv (int): Number of kv padding. Default: 1
        padding_q (int): Number of q padding. Default: 1
        with_cls_token (bool, optional):  If True, class token is included. Default: True
    """
    def __init__(self,
                 dim_in,
                 dim_out,
                 num_heads,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.,
                 method='dw_bn',
                 kernel_size=3,
                 stride_q=1,
                 stride_kv=1,
                 padding_q=1,
                 padding_kv=1,
                 with_cls_token=True,
                 **kwargs
                 ):
        super().__init__()
        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.dim = dim_out
        self.num_heads = num_heads
        # head_dim = self.qkv_dim // num_heads
        self.scale = dim_out ** -0.5
        self.with_cls_token = with_cls_token

        self.conv_proj_q = self._build_projection(
            dim_in, dim_out, kernel_size, padding_q,
            stride_q, 'linear' if method == 'avg' else method
        )
        self.conv_proj_k = self._build_projection(
            dim_in, dim_out, kernel_size, padding_kv,
            stride_kv, method
        )
        self.conv_proj_v = self._build_projection(
            dim_in, dim_out, kernel_size, padding_kv,
            stride_kv, method
        )

        self.proj_q = nn.Linear(dim_in, dim_out, bias_attr=qkv_bias)
        self.proj_k = nn.Linear(dim_in, dim_out, bias_attr=qkv_bias)
        self.proj_v = nn.Linear(dim_in, dim_out, bias_attr=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_out, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)

    def _build_projection(self,
                          dim_in,
                          dim_out,
                          kernel_size,
                          padding,
                          stride,
                          method):
        if method == 'dw_bn':
            proj = nn.Sequential(
                ('conv', nn.Conv2D(
                    dim_in,
                    dim_in,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    groups=dim_in,
                    bias_attr=False
                )),
                ('bn', nn.BatchNorm2D(dim_in)),
                ('rearrage', Rearrange('b c h w -> b (h w) c'))
            )
        elif method == 'dw_bn_new':
            proj = nn.Sequential(
                ('conv', DepthwiseSeparableConv2D(
                    dim_in,
                    dim_in,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    groups=dim_in,
                    bias_attr=False
                )),
                ('bn', nn.BatchNorm2D(dim_in)),
                ('rearrage', Rearrange('b c h w -> b (h w) c'))
            )
        elif method == 'avg':
            proj = nn.Sequential(
                ('avg', nn.AvgPool2D(
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    ceil_mode=True
                )),
                ('rearrage', Rearrange('b c h w -> b (h w) c')),
            )
        elif method == 'linear':
            proj = None
        else:
            raise ValueError('Unknown method ({})'.format(method))

        return proj

    def forward_conv(self, x, h, w):
        if self.with_cls_token:
            cls_token, x = paddle.split(x, [1, h * w], 1)

        B, L, C = x.shape
        assert L == h * w
        # rearrange x 'b (h w) c -> b c h w'
        x = x.transpose([0, 2, 1]).reshape([B, C, h, w])

        if self.conv_proj_q is not None:
            q = self.conv_proj_q(x)
        else:
            # rearrange x 'b c h w -> b (h w) c'
            q = x.flatten(2).transpose([0, 2, 1])

        if self.conv_proj_k is not None:
            k = self.conv_proj_k(x)
        else:
            # rearrange x 'b c h w -> b (h w) c'
            k = x.flatten(2).transpose([0, 2, 1])

        if self.conv_proj_v is not None:
            v = self.conv_proj_v(x)
        else:
            # rearrange x 'b c h w -> b (h w) c'
            v = x.flatten(2).transpose([0, 2, 1])

        if self.with_cls_token:
            q = paddle.concat((cls_token, q), axis=1)
            k = paddle.concat((cls_token, k), axis=1)
            v = paddle.concat((cls_token, v), axis=1)

        return q, k, v

    def forward(self, x, h, w):
        if (
            self.conv_proj_q is not None
            or self.conv_proj_k is not None
            or self.conv_proj_v is not None
        ):
            q, k, v = self.forward_conv(x, h, w)

        q = rearrange(self.proj_q(q), 'b l (h d) -> b h l d', h=self.num_heads)
        k = rearrange(self.proj_k(k), 'b l (h d) -> b h l d', h=self.num_heads)
        v = rearrange(self.proj_v(v), 'b l (h d) -> b h l d', h=self.num_heads)
        # print(f'q: {q.shape}, k: {k.shape}, v: {v.shape}')

        # paddle.einsum is only supported in dynamic graph mode
        # attn_score = paddle.einsum('bhlk,bhtk->bhlt', [q, k]) * self.scale
        attn_score = (q.matmul(k.transpose((0, 1, 3, 2)))) * self.scale
        # print(f'attn_score: {attn_score.shape}')
        attn = F.softmax(attn_score, axis=-1)
        attn = self.attn_drop(attn)
        # x = paddle.einsum('bhlt,bhtv->bhlv', [attn, v])
        x = attn.matmul(v)
        # print(f'x: {x.shape}')

        # rearrange x 'b h l d -> b l (h d)'
        x = x.transpose([0, 2, 1, 3]).flatten(2)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class TransformerBlock(nn.Layer):
    r""" Convolutional Transformer Block.

    Args:
        dim_in (int): Number of input channels.
        dim_out (int): Number of output channels.
        num_heads (int): Number of attention heads. Default: 4
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: False
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Layer, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Layer, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self,
                 dim_in,
                 dim_out,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 **kwargs):
        super().__init__()

        self.with_cls_token = kwargs['with_cls_token']

        self.norm1 = norm_layer(dim_in)
        self.attn = Attention(
            dim_in, dim_out, num_heads, qkv_bias, attn_drop, drop, **kwargs
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim_out)
        mlp_hidden_dim = int(dim_out * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim_out,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop
        )

    def forward(self, x, h, w):
        res = x

        x = self.norm1(x)
        attn = self.attn(x, h, w)
        x = res + self.drop_path(attn)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class ConvEmbed(nn.Layer):
    """ Image to Conv Embedding

    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        stride (int): Number of stride size. Default: 4.
        padding (int): Number of padding size. Default: 2.
        norm_layer (nn.Layer, optional): Normalization layer. Default: None
    """

    def __init__(self,
                 patch_size=7,
                 in_chans=3,
                 embed_dim=64,
                 stride=4,
                 padding=2,
                 norm_layer=None):
        super().__init__()

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2D(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=padding
        )
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        x = self.proj(x)

        B, C, H, W = x.shape
        # rearrange x 'b c h w -> b (h w) c'
        x = x.flatten(2).transpose([0, 2, 1])
        if self.norm is not None:
            x = self.norm(x)

        # rearrange x 'b (h w) c -> b c h w'
        x = x.transpose([0, 2, 1]).reshape([B, C, H, W])

        return x
    

class BasicBlock(nn.Layer):
    """ A basic Convolutional Transformer layer for one stage with support for patch or hybrid CNN input

    Args:
        patch_size (int | tuple(int)): Patch size. Default: 16
        patch_stride (int): Number of patch stride size. Default: 16
        patch_padding (int): Number of patch padding size. Default: 0
        in_chans (int): Number of input channels. Default: 3
        embed_dim (int): Number of input channels. Default: 768
        depth (int): Number of blocks.  Default: 12
        num_heads (int): Number of attention heads. Default: 12
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: False
        
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0
        act_layer (nn.Layer, optional): Active layer. Default: nn.GELU
        norm_layer (nn.Layer, optional): Normalization layer. Default: nn.LayerNorm
        init (str, optional): model init method, xavier and trunc_norm. Default: trunc_norm
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """
    def __init__(self,
                 patch_size=16,
                 patch_stride=16,
                 patch_padding=0,
                 in_chans=3,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 qkv_bias=False,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 init='trunc_norm',
                 **kwargs):
        super().__init__()
        # num_features for consistency with other models
        self.num_features = embed_dim
        self.embed_dim = embed_dim
        self.rearrage = None

        self.patch_embed = ConvEmbed(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            stride=patch_stride,
            padding=patch_padding,
            norm_layer=norm_layer
        )

        with_cls_token = kwargs['with_cls_token']
        if with_cls_token:
            self.cls_token = self.create_parameter(shape=(1, 1, embed_dim), default_initializer=zeros_)
            self.add_parameter("cls_token", self.cls_token)
        else:
            self.cls_token = None

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in paddle.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        blocks = []
        for j in range(depth):
            blocks.append(
                TransformerBlock(
                    dim_in=embed_dim,
                    dim_out=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    **kwargs
                )
            )
        self.blocks = nn.LayerList(blocks)

        if self.cls_token is not None:
            trunc_normal_(self.cls_token)

        if init == 'xavier':
            self.apply(self._init_weights_xavier)
        else:
            self.apply(self._init_weights_trunc_normal)

    def _init_weights_trunc_normal(self, m):
        if isinstance(m, nn.Linear):
            logging.info('=> init weight of Linear from trunc norm')
            trunc_normal_(m.weight)
            if m.bias is not None:
                logging.info('=> init bias of Linear to zeros')
                zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2D)):
            zeros_(m.bias)
            ones_(m.weight)

    def _init_weights_xavier(self, m):
        if isinstance(m, nn.Linear):
            logging.info('=> init weight of Linear from xavier uniform')
            xavier_uniform_(m.weight)
            if m.bias is not None:
                logging.info('=> init bias of Linear to zeros')
                zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2D)):
            zeros_(m.bias)
            ones_(m.weight)

    def forward(self, x):
        x = self.patch_embed(x)
        B, C, H, W = x.shape

        # rearrange x 'b c h w -> b (h w) c'
        x = x.flatten(2).transpose([0, 2, 1])

        cls_tokens = None
        if self.cls_token is not None:
            # stole cls_tokens impl from Phil Wang, thanks
            cls_tokens = self.cls_token.expand([B, -1, -1])
            x = paddle.concat((cls_tokens, x), axis=1)

        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            x = blk(x, H, W)

        if self.cls_token is not None:
            cls_tokens, x = paddle.split(x, [1, H * W], 1)
        
        # rearrange x 'b (h w) c -> b c h w'
        x = x.transpose([0, 2, 1]).reshape([B, C, H, W])
        
        return x, cls_tokens


class ConvTransformer(nn.Layer):
    """ Convolutional Transformer
        A PaddlePaddle impl of : `CvT: Introducing Convolutions to Vision Transformers`
         - https://arxiv.org/pdf/2103.15808

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        act_layer (nn.Layer): Activate layer. Default: nn.GELU.
        norm_layer (nn.Layer): Normalization layer. Default: nn.LayerNorm.
        init (str): init method. Default: trunc_norm.
        spec (dict): specific arguments. Default: None
    """
    def __init__(self,
                 in_chans=3,
                 class_num=1000,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 init='trunc_norm',
                 spec=None):
        super().__init__()
        self.class_num = class_num
        
        self.num_stages = spec.get('num_stages', 3)
        for i in range(self.num_stages):
            kwargs = {
                # BasicBlock
                'patch_size': spec['patch_size'][i],
                'patch_stride': spec['patch_stride'][i],
                'patch_padding': spec['patch_padding'][i],
                'embed_dim': spec['embed_dim'][i],
                'depth': spec['depth'][i],
                'num_heads': spec['num_heads'][i],
                'mlp_ratio': spec['mlp_ratio'][i],
                'drop_rate': spec['drop_rate'][i],
                'attn_drop_rate': spec['attn_drop_rate'][i],
                'drop_path_rate': spec['drop_path_rate'][i],
                'qkv_bias': spec['qkv_bias'][i],
                # Attention
                'method': spec['qkv_proj_method'][i],
                'kernel_size': spec['kernel_qkv'][i],
                'stride_q': spec['stride_q'][i],
                'stride_kv': spec['stride_kv'][i],
                'padding_q': spec['padding_q'][i],
                'padding_kv': spec['padding_kv'][i],
                'with_cls_token': spec['with_cls_token'][i],
            }

            stage = BasicBlock(
                in_chans=in_chans,
                act_layer=act_layer,
                norm_layer=norm_layer,
                init=init,
                **kwargs
            )
            setattr(self, f'stage{i}', stage)
            in_chans = spec['embed_dim'][i]

        dim_embed = spec['embed_dim'][-1]
        self.norm = norm_layer(dim_embed)
        self.cls_token = spec['with_cls_token'][-1]

        # Classifier head
        self.head = nn.Linear(dim_embed, class_num) if class_num > 0 else nn.Identity()
        trunc_normal_(self.head.weight)

    def forward_features(self, x):
        for i in range(self.num_stages):
            x, cls_tokens = getattr(self, f'stage{i}')(x)

        if self.cls_token:
            # [B, 1, D]
            x = self.norm(cls_tokens)
            x = paddle.squeeze(x, axis=1)
        else:
            # rearrange x 'b c h w -> b (h w) c'
            x = x.flatten(2).transpose([0, 2, 1])
            x = self.norm(x)
            x = paddle.mean(x, axis=1)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x


def _load_pretrained(pretrained, model, model_url, use_ssld=False):
    if pretrained is False:
        pass
    elif pretrained is True:
        load_dygraph_pretrain_from_url(model, model_url, use_ssld=use_ssld)
    elif isinstance(pretrained, str):
        load_dygraph_pretrain(model, pretrained)
    else:
        raise RuntimeError(
            "pretrained type is not available. Please use `string` or `boolean` type."
        )


def CvT_13_224(pretrained=False, use_ssld=False, **kwargs):
    spec = kwargs['spec']

    model = ConvTransformer(
        in_chans=3,
        class_num=kwargs['class_num'],
        act_layer=QuickGELU,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-5),
        init=spec.get('init', 'trunc_norm'),
        spec=spec
    )

    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["CvT_13_224"],
        use_ssld=use_ssld)

    return model


def CvT_13_384(pretrained=False, use_ssld=False, **kwargs):
    spec = kwargs['spec']

    model = ConvTransformer(
        in_chans=3,
        class_num=kwargs['class_num'],
        act_layer=QuickGELU,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-5),
        init=spec.get('init', 'trunc_norm'),
        spec=spec
    )

    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["CvT_13_384"],
        use_ssld=use_ssld)

    return model


def CvT_21_224(pretrained=False, use_ssld=False, **kwargs):
    spec = kwargs['spec']

    model = ConvTransformer(
        in_chans=3,
        class_num=kwargs['class_num'],
        act_layer=QuickGELU,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-5),
        init=spec.get('init', 'trunc_norm'),
        spec=spec
    )

    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["CvT_21_224"],
        use_ssld=use_ssld)

    return model


def CvT_21_384(pretrained=False, use_ssld=False, **kwargs):
    spec = kwargs['spec']

    model = ConvTransformer(
        in_chans=3,
        class_num=kwargs['class_num'],
        act_layer=QuickGELU,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-5),
        init=spec.get('init', 'trunc_norm'),
        spec=spec
    )

    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["CvT_21_384"],
        use_ssld=use_ssld)

    return model


def CvT_w24_384(pretrained=False, use_ssld=False, **kwargs):
    spec = kwargs['spec']

    model = ConvTransformer(
        in_chans=3,
        class_num=kwargs['class_num'],
        act_layer=QuickGELU,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-5),
        init=spec.get('init', 'trunc_norm'),
        spec=spec
    )

    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["CvT_w24_384"],
        use_ssld=use_ssld)
    
    return model
