import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import calculate_gain
from torch.nn.utils import spectral_norm

from utils import pixel_norm, resample_signal


class PixelNorm(nn.Module):
    def forward(self, x):
        return pixel_norm(x)


class ScaledTanh(nn.Tanh):
    def __init__(self, scale=0.5):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return super().forward(x * self.scale)


class GDropLayer(nn.Module):
    def __init__(self, strength=0.2, axes=(0, 1)):
        super().__init__()
        self.strength = strength
        self.axes = [axes] if isinstance(axes, int) else list(axes)

    def forward(self, x, deterministic=False):
        if deterministic or not self.strength:
            return x
        rnd_shape = [s if axis in self.axes else 1 for axis, s in enumerate(x.size())]
        rnd = (1 + self.strength) ** np.random.normal(size=rnd_shape)
        rnd = torch.from_numpy(rnd).type(x.data.type()).to(x)
        return x * rnd


class SelfAttention(nn.Module):
    def __init__(self, channels_in, sagan=True, spectral=True, factorized_attention=False, init='xavier_uniform'):
        super().__init__()
        d_key = max(channels_in // 8, 2)
        conv_conf = dict(kernel_size=1, equalized=False, spectral=spectral, init=init, bias=False, act_alpha=-1)
        self.gamma = nn.Parameter(torch.tensor(0.), requires_grad=True)
        self.pooling = nn.MaxPool1d(4) if sagan else nn.Sequential()
        self.key_conv = GeneralConv(channels_in, d_key, **conv_conf)
        self.query_conv = GeneralConv(channels_in, d_key, **conv_conf)
        self.value_conv = GeneralConv(channels_in, channels_in // 2, **conv_conf)
        self.final_conv = GeneralConv(channels_in // 2, channels_in, **conv_conf)
        self.softmax = nn.Softmax(dim=-1)
        self.scale = 1.0 if sagan else math.sqrt(d_key)
        self.factorized_attention = factorized_attention
        if spectral:
            self.key_conv = spectral_norm(self.key_conv)
            self.query_conv = spectral_norm(self.query_conv)
            self.value_conv = spectral_norm(self.value_conv)
            self.final_conv = spectral_norm(self.final_conv)

    def forward(self, x):  # BCT
        query = self.query_conv(x)  # BC/8T
        key = self.pooling(self.key_conv(x))  # BC/8T[/4]
        value = self.pooling(self.value_conv(x))  # BC/2T[/4]
        if not self.factorized_attention:
            out = F.softmax(torch.bmm(key.permute(0, 2, 1), query) / self.scale, dim=1)  # Bnormed(T[/4])T
            attention_map = out
            out = torch.bmm(value, out)  # BC/2T
        else:
            out = torch.bmm(value, F.softmax(key.permute(0, 2, 1), dim=2))  # BC/2C/8
            attention_map = None
            out = torch.bmm(out, F.softmax(query, dim=2)) / self.scale  # BC/2T
        out = self.final_conv(out)  # BCT
        return self.gamma * out + x, attention_map


class MinibatchStddev(nn.Module):
    def __init__(self, group_size=4):
        super().__init__()
        self.group_size = group_size if group_size != 0 else 1e6

    def forward(self, x):  # B, C, T
        s = x.size()
        group_size = min(s[0], self.group_size)
        y = x.view(group_size, -1, s[1], s[2])  # G,B//G,C,T
        y = y - y.mean(dim=0, keepdim=True)  # G,B//G,C,T
        y = torch.sqrt((y ** 2).mean(dim=0))  # B//G,C,T
        y = y.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)  # B//G,1,1
        y = y.repeat((group_size, 1, s[2]))  # B,1,T
        return torch.cat([x, y], dim=1)


class ConditionalBatchNorm(nn.Module):
    def __init__(self, num_features, num_classes, spectral, latent_size=0):
        super().__init__()
        self.no_cond = latent_size == 0 and num_classes == 0
        self.num_features = num_features
        self.normalizer = nn.BatchNorm1d(num_features, affine=self.no_cond)
        if self.no_cond:
            return
        if latent_size == 0:
            self.embed = GeneralConv(num_classes, num_features * 2, kernel_size=1, equalized=False, act_alpha=-1,
                                     spectral=spectral, bias=False)
            self.mode = 'CBN'
        else:
            self.embed = nn.Sequential(
                GeneralConv(latent_size, num_features * 4, kernel_size=1, equalized=False, act_alpha=0.0, spectral=True,
                            bias=True),
                GeneralConv(num_features * 4, num_features * 2, kernel_size=1, equalized=False, act_alpha=-1,
                            spectral=spectral, bias=False))
            if num_classes == 0:
                self.mode = 'SM'
            else:
                self.embed_add = GeneralConv(num_classes, 2 * latent_size, kernel_size=1, equalized=False, act_alpha=-1,
                                             spectral=spectral, bias=False)
                self.mode = 'CSM'

    def forward(self, x, y, z=None):  # y is B, num_classes, Ty and x is B, num_features, Tx
        out = self.normalizer(x)
        if self.no_cond:
            return out
        if self.mode == 'CBN':
            cond = y
        else:
            if self.mode == 'SM':
                cond = z.unsqueeze(2)
            else:
                add, mul = self.embed_add(y).chunk(2, dim=1)
                cond = z.unsqueeze(2).repeat(-1, -1, add.size(2))
                cond = cond + add + cond * mul
        embed = self.embed(cond)  # B, num_features*2, Ty
        embed = resample_signal(embed, embed.shape[2], x.shape[2], pytorch=True)
        gamma, beta = embed.chunk(2, dim=1)
        return out + gamma * out + beta  # trick to make sure gamma is 1.0 at the beginning of the training


class EqualizedSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, spectral=False,
                 equalized=True, init='kaiming_normal', act_alpha=0.0, bias=True):
        super().__init__()
        self.net = nn.Sequential(
            EqualizedConv1d(in_channels, in_channels, kernel_size, padding, spectral, equalized, init, act_alpha, True,
                            groups=in_channels),
            EqualizedConv1d(in_channels, out_channels, 1, 0, spectral, equalized, init, act_alpha, bias))

    def forward(self, x):
        return self.net(x)


class EqualizedConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, spectral=False,
                 equalized=True, init='kaiming_normal', act_alpha=0.0, bias=True, groups=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, padding=padding, bias=bias, groups=groups)
        if bias:
            self.conv.bias.data.zero_()
        act_alpha = act_alpha if act_alpha > 0 else 1
        if init == 'kaiming_normal':
            torch.nn.init.kaiming_normal_(self.conv.weight, a=act_alpha)
        elif init == 'xavier_uniform':
            torch.nn.init.xavier_uniform_(self.conv.weight, gain=calculate_gain('leaky_relu', param=act_alpha))
        elif init == 'orthogonal':
            torch.nn.init.orthogonal_(self.conv.weight, gain=calculate_gain('leaky_relu', param=act_alpha))
        if not equalized:
            self.scale = 1.0
        else:
            self.scale = ((torch.mean(self.conv.weight.data ** 2)) ** 0.5).item()
        self.conv.weight.data.copy_(self.conv.weight.data / self.scale)
        if spectral:
            self.conv = spectral_norm(self.conv)

    def forward(self, x):
        return self.conv(x * self.scale)


# TODO add separable flag to G and D
class GeneralConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, equalized=True, pad=None, act_alpha=0.0, do=0,
                 num_classes=0, act_norm=None, spectral=False, init='kaiming_normal', bias=True, separable=False):
        super().__init__()
        pad = (kernel_size - 1) // 2 if pad is None else pad
        if separable:
            conv_class = EqualizedSeparableConv1d
        else:
            conv_class = EqualizedConv1d
        conv = conv_class(in_channels, out_channels, kernel_size, padding=pad, spectral=spectral,
                          equalized=equalized, init=init, act_alpha=act_alpha,
                          bias=bias if act_norm != 'batch' else False)
        norm = None
        if act_norm == 'batch':
            # TODO add latent_size based on CBN or SM or CSM
            norm = ConditionalBatchNorm(out_channels, num_classes, spectral)
        self.conv = conv
        self.norm = norm
        self.net = []
        if act_alpha >= 0:
            if act_alpha == 0:
                self.net.append(nn.ReLU())  # DO NOT use inplace, gradient penalty will break
            else:
                self.net.append(nn.LeakyReLU(act_alpha))  # DO NOT use inplace, gradient penalty will break
        if act_norm == 'pixel':
            self.net.append(PixelNorm())
        if do != 0:
            self.net.append(GDropLayer(strength=do))
        self.net = nn.Sequential(*self.net)

    def forward(self, x, y=None, z=None, *args, **kwargs):
        c = self.conv(x)
        if z is not None:
            c = c * z
        if self.norm:
            c = self.norm(c, y)
        return self.net(c)


class PassChannelResidual(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        if x.size(1) >= y.size(1):
            x[:, :y.size(1)] = x[:, :y.size(1)] + y
            return x
        else:
            y[:, :x.size(1)] = y[:, :x.size(1)] + x
            return y


class ConcatResidual(nn.Module):
    def __init__(self, ch_in, ch_out, equalized, spectral, init):
        super().__init__()
        self.net = GeneralConv(ch_in, ch_out - ch_in, kernel_size=1,
                               equalized=equalized, act_alpha=-1, spectral=spectral, init=init)

    def forward(self, x, h):
        return h + torch.cat([x, self.net(x)], dim=1)
