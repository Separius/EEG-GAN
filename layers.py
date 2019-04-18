import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import calculate_gain
from torch.nn.utils import spectral_norm

from utils import pixel_norm, resample_signal, expand3d


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
    def __init__(self, channels_in, spectral, init='xavier_uniform'):
        super().__init__()
        d_key = max(channels_in // 8, 2)
        conv_conf = dict(kernel_size=1, equalized=False, spectral=spectral,
                         init=init, bias=False, act_alpha=-1)
        self.gamma = nn.Parameter(torch.tensor(0.), requires_grad=True)
        self.pooling = nn.MaxPool1d(4)
        self.key_conv = GeneralConv(channels_in, d_key, **conv_conf)
        self.query_conv = GeneralConv(channels_in, d_key, **conv_conf)
        self.value_conv = GeneralConv(channels_in, channels_in // 2, **conv_conf)
        self.final_conv = GeneralConv(channels_in // 2, channels_in, **conv_conf)
        self.softmax = nn.Softmax(dim=-1)
        self.scale = 1.0

    def forward(self, x):  # BCT
        query = self.query_conv(x)  # BC/8T
        key = self.pooling(self.key_conv(x))  # BC/8T[/4]
        value = self.pooling(self.value_conv(x))  # BC/2T[/4]
        out = F.softmax(torch.bmm(key.permute(0, 2, 1), query) / self.scale, dim=1)  # Bnormed(T[/4])T
        attention_map = out
        out = torch.bmm(value, out)  # BC/2T
        out = self.final_conv(out)  # BCT
        return self.gamma * out + x, attention_map


class MinibatchStddev(nn.Module):
    def __init__(self, group_size=4, temporal_groups_per_window=1, kernel_size=32):
        super().__init__()
        self.group_size = group_size if group_size != 0 else 1e6
        self.kernel_size = kernel_size
        self.stride_size = self.kernel_size // temporal_groups_per_window

    def forward(self, x):  # B, C, T
        s = x.size()
        group_size = min(s[0], self.group_size)
        all_y = []
        for i in range(s[2] // self.stride_size):
            y = x[..., i * self.stride_size:(i + 1) * self.stride_size]
            y = y.view(group_size, -1, s[1], self.stride_size)  # G,B//G,C,T
            y = y - y.mean(dim=0, keepdim=True)  # G,B//G,C,T
            y = torch.sqrt((y ** 2).mean(dim=0))  # B//G,C,T
            y = y.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)  # B//G,1,1
            y = y.repeat((group_size, 1, self.stride_size))  # B,1,T
            all_y.append(y)
        return torch.cat([x, torch.cat(all_y, dim=2)], dim=1)


class ConditionalBatchNorm(nn.Module):
    def __init__(self, num_features, num_classes, latent_size, spectral):
        super().__init__()
        no_cond = latent_size == 0 and num_classes == 0
        self.num_features = num_features
        self.normalizer = nn.BatchNorm1d(num_features, affine=no_cond)
        if no_cond:
            self.mode = 'BN'  # batch norm
        elif latent_size == 0:
            self.embed = GeneralConv(num_classes, num_features * 2, kernel_size=1, equalized=False,
                                     act_alpha=-1, spectral=spectral, bias=False)
            self.mode = 'CBN'  # conditional batch norm
        else:  # both 'SM'(self modulation) and 'CSM'(conditional self modulation)
            # NOTE maybe reduce it to a single layer linear network?
            self.embed = nn.Sequential(
                GeneralConv(latent_size + num_classes, num_features * 2, kernel_size=1,
                            equalized=False, act_alpha=0.0, spectral=spectral, bias=True),
                GeneralConv(num_features * 2, num_features * 2, kernel_size=1,
                            equalized=False, act_alpha=-1, spectral=spectral, bias=False))
            if num_classes == 0:
                self.mode = 'SM'  # self modulation
            else:
                self.mode = 'CSM'  # conditional self modulation(biggan)

    def forward(self, x, y, z):  # y = B*num_classes*Ty ; x = B*num_features*Tx ; z = B*latent_size
        out = self.normalizer(x)
        if self.mode == 'BN':
            return out
        if y is not None and y.ndimension() == 2:
            y = y.unsqueeze(2)
        if self.mode == 'CBN':
            cond = y
        else:
            if self.mode == 'CSM':
                z = expand3d(z)
                cond = torch.cat([resample_signal(z, z.size(2), y.size(2), pytorch=True), y], dim=1)
            else:
                cond = expand3d(z)
        embed = self.embed(cond)  # B, num_features*2, Ty
        embed = resample_signal(embed, embed.shape[2], out.shape[2], pytorch=True)
        gamma, beta = embed.chunk(2, dim=1)
        return out + gamma * out + beta  # trick to make sure gamma is 1.0 at the beginning of the training


class EqualizedSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 padding, spectral, equalized, init, act_alpha, bias, groups, stride):
        super().__init__()
        self.net = nn.Sequential(
            EqualizedConv1d(in_channels, in_channels, kernel_size, padding,
                            spectral, equalized, init, -1, True, groups=in_channels, stride=stride),
            EqualizedConv1d(in_channels, out_channels, 1, 0, spectral, equalized, init, act_alpha, bias, groups, 1))

    def forward(self, x):
        return self.net(x)


class EqualizedConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding,
                 spectral, equalized, init, act_alpha, bias, groups, stride):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, stride=stride,
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


class GeneralConv(nn.Module):
    def __init__(self, in_channels, out_channels, z_to_bn_size=0, kernel_size=3, equalized=True,
                 pad=None, act_alpha=0.2, do=0, num_classes=0, act_norm=None, spectral=False,
                 init='kaiming_normal', bias=True, separable=False, stride=1):
        super().__init__()
        pad = (kernel_size - 1) // 2 if pad is None else pad
        if separable:
            conv_class = EqualizedSeparableConv1d
        else:
            conv_class = EqualizedConv1d
        conv = conv_class(in_channels, out_channels, kernel_size, padding=pad, spectral=spectral,
                          equalized=equalized, init=init, act_alpha=act_alpha,
                          bias=bias if act_norm != 'batch' else False, groups=1, stride=stride)
        norm = None
        if act_norm == 'batch':
            norm = ConditionalBatchNorm(out_channels, num_classes, z_to_bn_size, spectral)
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

    def forward(self, x, y=None, z=None, conv_noise=None):
        c = self.conv(x)
        if conv_noise is not None:
            c = c * conv_noise
        if self.norm:
            c = self.norm(c, y, z)
        return self.net(c)


class PassChannelResidual(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        if x.size(1) >= y.size(1):
            x[:, :y.size(1)] = x[:, :y.size(1)] + y
            return x
        return y[:, :x.size(1)] + x


class ConcatResidual(nn.Module):
    def __init__(self, ch_in, ch_out, equalized, spectral, init):
        super().__init__()
        assert ch_out >= ch_in
        if ch_out > ch_in:
            self.net = GeneralConv(ch_in, ch_out - ch_in, kernel_size=1, equalized=equalized, act_alpha=-1,
                                   spectral=spectral, init=init)
        else:
            self.net = None

    def forward(self, h, x):
        if self.net:
            return h + torch.cat([x, self.net(x)], dim=1)
        return h + x

