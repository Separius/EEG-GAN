import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import calculate_gain
from torch.nn.utils import spectral_norm

from utils import cudize, pixel_norm, resample_signal


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
        rnd = cudize(torch.from_numpy(rnd).type(x.data.type()))
        return x * rnd


class SelfAttention(nn.Module):
    def __init__(self, channels_in, sagan=True, spectral=True, factorized_attention=False, init='xavier_uniform'):
        super().__init__()
        d_key = max(channels_in // 8, 2)
        conv_conf = dict(kernel_size=1, equalized=False, spectral=spectral, init=init, bias=False, act_alpha=-1)
        self.gamma = 0
        self.pooling = nn.MaxPool1d(2) if sagan else nn.Sequential()
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
        if self.gamma == 0:
            return x, None
        query = self.query_conv(x)  # BC/8T
        key = self.pooling(self.key_conv(x))  # BC/8T[/2]
        value = self.pooling(self.value_conv(x))  # BC/2T[/2]
        if not self.factorized_attention:
            out = F.softmax(torch.bmm(key.permute(0, 2, 1), query) / self.scale, dim=1)  # Bnormed(T[/2])T
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
    def __init__(self, num_features, num_classes, spectral):
        super().__init__()
        self.num_features = num_features
        self.normalizer = nn.BatchNorm1d(num_features, affine=False)
        self.embed = GeneralConv(num_classes, num_features * 2, kernel_size=1, equalized=False, act_alpha=-1,
                                 spectral=spectral, bias=False)

    def forward(self, x, y):  # y is B, num_classes, Ty and x is B, num_features, Tx
        out = self.normalizer(x)
        embed = self.embed(y)  # B, num_features*2, Ty
        embed = resample_signal(embed, embed.shape[2], x.shape[2], pytorch=True)
        gamma, beta = embed.chunk(2, dim=1)
        return gamma * out + beta


class EqualizedConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0,
                 spectral=False, equalized=True, init='kaiming_normal', act_alpha=0.0, bias=True):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, padding=padding, bias=bias)
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


class GeneralConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, equalized=True, pad=None, act_alpha=0.0, do=0,
                 num_classes=0, act_norm=None, spectral=False, init='kaiming_normal', bias=True):
        super().__init__()
        pad = (kernel_size - 1) // 2 if pad is None else pad
        conv = EqualizedConv1d(in_channels, out_channels, kernel_size, padding=pad, spectral=spectral,
                               equalized=equalized, init=init, act_alpha=act_alpha,
                               bias=bias if act_norm != 'batch' else False)
        norm = None
        if act_norm == 'batch':
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
