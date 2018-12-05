import math
import torch
import numpy as np
from torch import nn
from utils import cudize, pixel_norm
from torch.nn.init import calculate_gain
from torch.nn.utils import spectral_norm


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return pixel_norm(x)


class ScaledTanh(nn.Tanh):
    def __init__(self, scale=0.5):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return super().forward(x * self.scale)


class GDropLayer(nn.Module):
    """
    # Generalized dropout layer. Supports arbitrary subsets of axes and different
    # modes. Mainly used to inject multiplicative Gaussian noise in the network.
    """

    def __init__(self, mode='mul', strength=0.2, axes=(0, 1), normalize=False):
        super().__init__()
        self.mode = mode.lower()
        assert self.mode in {'mul', 'drop', 'prop'}, 'Invalid GDropLayer mode' % mode
        self.strength = strength
        self.axes = [axes] if isinstance(axes, int) else list(axes)
        self.normalize = normalize

    def forward(self, x, deterministic=False):
        if deterministic or not self.strength:
            return x

        rnd_shape = [s if axis in self.axes else 1 for axis, s in enumerate(x.size())]
        if self.mode == 'drop':
            p = 1 - self.strength
            rnd = np.random.binomial(1, p=p, size=rnd_shape) / p
        elif self.mode == 'mul':
            rnd = (1 + self.strength) ** np.random.normal(size=rnd_shape)
        else:
            coef = self.strength * x.size(1) ** 0.5
            rnd = np.random.normal(size=rnd_shape) * coef + 1

        if self.normalize:
            rnd = rnd / np.linalg.norm(rnd, keepdims=True)
        rnd = cudize(torch.autograd.Variable(torch.from_numpy(rnd).type(x.data.type())))
        return x * rnd

    def __repr__(self):
        param_str = '(mode = %s, strength = %s, axes = %s, normalize = %s)' % (
            self.mode, self.strength, self.axes, self.normalize)
        return self.__class__.__name__ + param_str


class SelfAttention(nn.Module):
    def __init__(self, channels_in, spectral=True):
        super().__init__()
        d_key = max(channels_in // 8, 2)
        self.gamma = 0
        self.key_conv = nn.Conv1d(channels_in, d_key, kernel_size=1, bias=False)
        self.query_conv = nn.Conv1d(channels_in, d_key, kernel_size=1, bias=False)
        self.value_conv = nn.Conv1d(channels_in, channels_in, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.scale = math.sqrt(d_key)
        if spectral:
            self.value_conv = spectral_norm(self.value_conv)

    def forward(self, x):
        if self.gamma == 0:
            return x
        batch_size, _, t = x.size()
        query = self.query_conv(x).permute(0, 2, 1)
        key = self.key_conv(x)
        energy = torch.bmm(query, key)
        attention = self.softmax(energy / self.scale)
        value = self.value_conv(x)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, -1, t)
        return self.gamma * out + x


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


class ConditionalGeneralNorm(nn.Module):
    def __init__(self, num_features, num_classes, norm_class):
        super().__init__()
        if num_classes == 0:
            self.normalizer = norm_class(num_features)
            self.embed = None
        else:
            self.num_features = num_features
            self.normalizer = norm_class(num_features, affine=False)
            self.embed = nn.Embedding(num_classes, num_features * 2)
            self.embed.weight.data[:, :num_features].normal_(1, 0.02)
            self.embed.weight.data[:, num_features:].zero_()

    def forward(self, x, y=None):
        out = self.normalizer(x)
        if self.embed is None or y is None:
            return out
        gamma, beta = self.embed(y).chunk(2, dim=1)
        return gamma.view(-1, self.num_features, 1) * out + beta.view(-1, self.num_features, 1)


class ConditionalBatchNorm(ConditionalGeneralNorm):
    def __init__(self, num_features, num_classes):
        super().__init__(num_features, num_classes, nn.BatchNorm1d)


class ConditionalLayerNorm(ConditionalGeneralNorm):
    def __init__(self, num_features, num_classes):
        def norm_class(*args, **kwargs):
            return nn.GroupNorm(1, *args, **kwargs)

        super().__init__(num_features, num_classes, norm_class)


class EqualizedConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, spectral=False, equalized=True,
                 init='kaiming_normal'):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, padding=padding, bias=True)
        self.conv.bias.data.zero_()
        if init == 'kaiming_normal':
            torch.nn.init.kaiming_normal_(self.conv.weight, a=calculate_gain('conv1d'))
        elif init == 'xavier_uniform':
            torch.nn.init.xavier_uniform_(self.conv.weight, math.sqrt(2))
        elif init == 'orthogonal':
            torch.nn.init.orthogonal_(self.conv.weight)
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
    def __init__(self, in_channels, out_channels, kernel_size=3, equalized=True, pad=None, act_alpha=0, do=0,
                 do_mode='mul', num_classes=0, act_norm=None, spectral=False, init='kaiming_normal'):
        super().__init__()
        pad = (kernel_size - 1) // 2 if pad is None else pad
        conv = EqualizedConv1d(in_channels, out_channels, kernel_size, padding=pad, spectral=spectral,
                               equalized=equalized, init=init)
        norm = None
        if act_norm == 'layer':
            norm = ConditionalLayerNorm(out_channels, num_classes)
        elif act_norm == 'batch':
            norm = ConditionalBatchNorm(out_channels, num_classes)
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
            self.net.append(GDropLayer(strength=do, mode=do_mode))
        self.net = nn.Sequential(*self.net)

    def forward(self, x, y=None, *args, **kwargs):
        if self.norm:
            return self.net(self.norm(self.conv(x), y))
        return self.net(self.conv(x))

    def __repr__(self):
        param_str = '(conv = {}, norm = {})'.format(self.conv.conv, self.norm)
        return self.__class__.__name__ + param_str
