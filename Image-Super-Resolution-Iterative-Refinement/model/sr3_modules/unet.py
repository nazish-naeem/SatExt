import math
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction
from model.layers import GBlock
from model import layers
import functools


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

# PositionalEncoding Source： https://github.com/lmnt-com/wavegrad/blob/master/src/wavegrad/model.py
class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(
            1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels*(1+self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        # print(x.dtype)
        # print(noise_embed.dtype)
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1)
        return x


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0, use_affine_level=False, norm_groups=32):
        super().__init__()
        self.noise_func = FeatureWiseAffine(
            noise_level_emb_dim, dim_out, use_affine_level)

        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        # print(x.dtype)
        b, c, h, w = x.shape
        h = self.block1(x)
        h = self.noise_func(h, time_emb)
        h = self.block2(h)
        return h + self.res_conv(x)


class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        # print(input.dtype)
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input


class ResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, *, noise_level_emb_dim=None, norm_groups=32, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, noise_level_emb_dim, norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            self.attn = SelfAttention(dim_out, norm_groups=norm_groups)

    def forward(self, x, time_emb):
        x = self.res_block(x, time_emb)
        if(self.with_attn):
            x = self.attn(x)
        return x

import time as tm

class UNet(nn.Module):
    def __init__(
        self,
        in_channel=6,
        out_channel=3,
        inner_channel=32,
        norm_groups=32,
        channel_mults=(1, 2, 4, 8, 8),
        attn_res=(8),
        res_blocks=3,
        dropout=0,
        with_noise_level_emb=True,
        image_size=128
    ):
        super().__init__()
        

        if with_noise_level_emb:
            noise_level_channel = inner_channel
            self.noise_level_mlp = nn.Sequential(
                PositionalEncoding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel)
            )
        else:
            noise_level_channel = None
            self.noise_level_mlp = None
        self.scale = 0.707
        self.activation = nn.ReLU(inplace=False)
        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size
        downs = [nn.Conv2d(in_channel, inner_channel,
                           kernel_size=3, padding=1)]
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            self.which_conv = functools.partial(layers.SNConv2d,
                          kernel_size=3, padding=1,
                          num_svs=1, num_itrs=1,
                          eps=1e-12)
            self.which_linear = functools.partial(layers.SNLinear,
                          num_svs=1, num_itrs=1,
                          eps=1e-12)
            # self.which_conv = functools.partial(nn.Conv2d, kernel_size=3, padding=1)
            # self.which_linear = nn.Linear
            bn_linear = functools.partial(self.which_linear, bias=False)
            self.which_bn = functools.partial(layers.ccbn,
                            which_linear=bn_linear,
                            cross_replica=False,
                            mybn=False,
                            input_size=pre_channel,
                            norm_style='bn',
                            eps=1e-5)
            
            for _ in range(0, res_blocks):
                # print('prechannel and channel mult')
                # print(pre_channel)
                # print(channel_mult)
                # downs.append(ResnetBlocWithAttn(
                #     pre_channel, channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups, dropout=dropout, with_attn=use_attn))
                downs.append(GBlock(
                    pre_channel, channel_mult,which_conv=self.which_conv, which_bn=self.which_bn, activation=self.activation))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res//2
        self.downs = nn.ModuleList(downs)

        # self.mid = nn.ModuleList([
        #     ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
        #                        dropout=dropout, with_attn=True),
        #     ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
        #                        dropout=dropout, with_attn=False)
        # ])
        self.mid = nn.ModuleList([
            GBlock(pre_channel, pre_channel,which_conv=self.which_conv, which_bn=self.which_bn, activation=self.activation),
            GBlock(pre_channel, pre_channel,which_conv=self.which_conv, which_bn=self.which_bn, activation=self.activation)
        ])

        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks+1):
                xx = pre_channel+feat_channels.pop()
                which_bn = functools.partial(layers.ccbn,
                          which_linear=nn.Embedding,
                          cross_replica=False,
                          mybn=False,
                          input_size=xx,
                          norm_style='bn',
                          eps=1e-5)
                ups.append(GBlock(
                    xx,channel_mult,which_conv=self.which_conv, which_bn=self.which_bn, activation=self.activation))
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(pre_channel))
                now_res = now_res*2

        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(pre_channel, default(out_channel, in_channel), groups=norm_groups)

    def forward(self, x, time):
        # print(x.dtype)
        t = self.noise_level_mlp(time) if exists(
            self.noise_level_mlp) else None

        feats = []
        for layer in self.downs:
            count=0
            if isinstance(layer, GBlock):
                # print('layer gblock down')
                # print('down resnest')
                # print('x==',x.shape)
                # print('t==',t.shape)
                # x = layer(x, t)
                x = layer(x, t)
                # x = layer(t, x)
                # print('output==',x.shape)
                # tm.sleep(5)
                # count=count+1
                # if count==2:
                #     print(dd)
                #     dd=1
            else:
                # print('layer down')
                x = layer(x)
            feats.append(x)
        # print('down end')

        for layer in self.mid:
            
            if isinstance(layer, GBlock):
                # print('layer gblock mid')
                x = layer(x, t)
            else:
                # print('layer down')
                x = layer(x)

        for layer in self.ups:
            if isinstance(layer, GBlock):
                # x = layer(torch.cat((x, feats.pop()), dim=1), t)
                x = layer(torch.cat((x, feats.pop()*self.scale), dim=1), t)
            else:
                x = layer(x)

        return self.final_conv(x)


# class UNet(nn.Module):
#     def __init__(
#         self,
#         in_channel=6,
#         out_channel=3,
#         inner_channel=32,
#         norm_groups=32,
#         channel_mults=(1, 2, 4, 8, 8),
#         attn_res=(8),
#         res_blocks=3,
#         dropout=0,
#         with_noise_level_emb=True,
#         image_size=128
#     ):
#         super().__init__()

#         if with_noise_level_emb:
#             noise_level_channel = inner_channel
#             self.noise_level_mlp = nn.Sequential(
#                 PositionalEncoding(inner_channel),
#                 nn.Linear(inner_channel, inner_channel * 4),
#                 Swish(),
#                 nn.Linear(inner_channel * 4, inner_channel)
#             )
#         else:
#             noise_level_channel = None
#             self.noise_level_mlp = None
#         self.scale = 0.707
#         num_mults = len(channel_mults)
#         pre_channel = inner_channel
#         feat_channels = [pre_channel]
#         now_res = image_size
#         downs = [nn.Conv2d(in_channel, inner_channel,
#                            kernel_size=3, padding=1)]
#         for ind in range(num_mults):
#             is_last = (ind == num_mults - 1)
#             use_attn = (now_res in attn_res)
#             channel_mult = inner_channel * channel_mults[ind]
#             for _ in range(0, res_blocks):
#                 downs.append(ResnetBlocWithAttn(
#                     pre_channel, channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups, dropout=dropout, with_attn=use_attn))
#                 feat_channels.append(channel_mult)
#                 pre_channel = channel_mult
#             if not is_last:
#                 downs.append(Downsample(pre_channel))
#                 feat_channels.append(pre_channel)
#                 now_res = now_res//2
#         self.downs = nn.ModuleList(downs)

#         self.mid = nn.ModuleList([
#             ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
#                                dropout=dropout, with_attn=True),
#             ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
#                                dropout=dropout, with_attn=False)
#         ])

#         ups = []
#         for ind in reversed(range(num_mults)):
#             is_last = (ind < 1)
#             use_attn = (now_res in attn_res)
#             channel_mult = inner_channel * channel_mults[ind]
#             for _ in range(0, res_blocks+1):
#                 ups.append(ResnetBlocWithAttn(
#                     pre_channel+feat_channels.pop(), channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
#                         dropout=dropout, with_attn=use_attn))
#                 pre_channel = channel_mult
#             if not is_last:
#                 ups.append(Upsample(pre_channel))
#                 now_res = now_res*2

#         self.ups = nn.ModuleList(ups)

#         self.final_conv = Block(pre_channel, default(out_channel, in_channel), groups=norm_groups)

#     def forward(self, x, time):
#         # print(x.dtype)
#         t = self.noise_level_mlp(time) if exists(
#             self.noise_level_mlp) else None

#         feats = []
#         for layer in self.downs:
#             if isinstance(layer, ResnetBlocWithAttn):
#                 print('down resnest')
#                 print('x==',x.shape)
#                 print('t==',t.shape)
#                 x = layer(x, t)
#                 print('output==',x.shape)
#                 tm.sleep(5)
#             else:
#                 x = layer(x)
#             feats.append(x)

#             # tm.sleep(1000)

#         for layer in self.mid:
#             if isinstance(layer, ResnetBlocWithAttn):
#                 # print('mid resnest')
#                 # print('x==',x.shape)
#                 # print('t==',t.shape)
#                 x = layer(x, t)
#                 # print('output==',x.shape)
#             else:
#                 x = layer(x)

#         for layer in self.ups:
#             if isinstance(layer, ResnetBlocWithAttn):
#                 # print('up resnest')
#                 # print('x==',x.shape)
#                 xx=feats.pop();
#                 # print('feats.pop()==',xx.shape)
#                 # x = layer(torch.cat((x, feats.pop()), dim=1), t)
#                 x = layer(torch.cat((x, xx*self.scale), dim=1), t)
#             else:
#                 x = layer(x)

#         return self.final_conv(x)
