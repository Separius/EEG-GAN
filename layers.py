import math
import torch
import numpy as np
from torch import nn
from torch.nn.utils import spectral_norm
from utils import cudize, pixel_norm, EPSILON
from torch.nn.init import calculate_gain, _calculate_correct_fan


class PixelNorm(nn.Module):
    def __init__(self):
        super(PixelNorm, self).__init__()

    def forward(self, x):
        return pixel_norm(x)


class ScaledTanh(nn.Tanh):
    def __init__(self, scale=0.5):
        super(ScaledTanh, self).__init__()
        self.scale = scale

    def forward(self, x):
        return super().forward(x * self.scale)


class GDropLayer(nn.Module):
    """
    # Generalized dropout layer. Supports arbitrary subsets of axes and different
    # modes. Mainly used to inject multiplicative Gaussian noise in the network.
    """

    def __init__(self, mode='mul', strength=0.2, axes=(0, 1), normalize=False):
        super(GDropLayer, self).__init__()
        self.mode = mode.lower()
        assert self.mode in ['mul', 'drop', 'prop'], 'Invalid GDropLayer mode' % mode
        self.strength = strength
        self.axes = [axes] if isinstance(axes, int) else list(axes)
        self.normalize = normalize
        self.gain = None

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
        super(SelfAttention, self).__init__()
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
        out = self.gamma * out + x
        return out


class MinibatchStddev(nn.Module):
    def __init__(self, group_size=4):
        super().__init__()
        self.group_size = group_size if group_size != 0 else 1e6

    def forward(self, x):  # B, C, T
        s = x.size()
        group_size = min(s[0], self.group_size)
        y = x.view(group_size, -1, s[1], s[2])  # G,B//G,C,T
        y = y - y.mean(dim=0, keepdim=True)  # G,B//G,C,T
        y = (y ** 2).mean(dim=0)  # B//G,C,T
        y = torch.sqrt(y + EPSILON)  # B//G,C,T
        y = y.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)  # B//G,1,1
        y = y.repeat((group_size, 1, s[2]))  # B,1,T
        return torch.cat([x, y], dim=1)


class MinibatchStddevOld(nn.Module):
    def __init__(self, group_size=4):
        super().__init__()
        self.temporal = False
        self.out_channels = 1

    def calc_mean(self, x, expand=True):
        mean = torch.mean(x, dim=0, keepdim=True)
        if not self.temporal:
            mean = torch.mean(mean, dim=2, keepdim=True)
        c = mean.size(1)
        if self.out_channels == c:
            return mean
        if self.out_channels == 1:
            return torch.mean(mean, dim=1, keepdim=True)
        else:
            step = c // self.out_channels
            if expand:
                return torch.cat(
                    [torch.mean(mean[:, step * i:step * (i + 1), :], dim=1, keepdim=True).expand(-1, step, -1) for i in
                     range(self.out_channels)], dim=1)
            return torch.cat([torch.mean(mean[:, step * i:step * (i + 1), :], dim=1, keepdim=True) for i in
                              range(self.out_channels)], dim=1)

    def forward(self, x):
        mean = self.calc_mean(x).expand(x.size())
        y = torch.sqrt(self.calc_mean((x - mean) ** 2, False)).expand(x.size(0), -1, x.size(2))
        return torch.cat((x, y), dim=1)


class ConditionalBatchNorm(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(ConditionalBatchNorm, self).__init__()
        if num_classes == 0:
            self.bn = nn.BatchNorm1d(num_channels)
        else:
            self.gamma_embedding = nn.EmbeddingBag(num_classes, num_channels)
            self.gamma_embedding.weight.data.fill_(1.0)
            self.beta_embedding = nn.EmbeddingBag(num_classes, num_channels)
            self.beta_embedding.weight.data.zero_()
            self.bn = None

    def forward(self, x, y=None):
        if self.bn is not None:
            return self.bn(x)
        x_size = x.size()
        channels = x_size[1]
        gammas = self.gamma_embedding(y).unsqueeze(2).expand(x_size)
        betas = self.beta_embedding(y).unsqueeze(2).expand(x_size)
        input_channel_major = x.permute(1, 0, 2).contiguous().view(channels, -1)
        mean = input_channel_major.mean(dim=1)
        var = input_channel_major.var(dim=1)
        x = (x - mean.view(1, channels, 1).expand(x_size)) * torch.rsqrt(
            var.view(1, channels, 1).expand(x_size) + EPSILON)
        return gammas * x + betas


class ConditionalLayerNorm(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(ConditionalLayerNorm, self).__init__()
        if num_classes == 0:
            self.gn = nn.GroupNorm(1, num_channels)
            return
        self.gamma_embedding = nn.EmbeddingBag(num_classes, num_channels)
        self.gamma_embedding.weight.data.fill_(1.0)
        self.beta_embedding = nn.EmbeddingBag(num_classes, num_channels)
        self.beta_embedding.weight.data.zero_()
        self.gn = None

    def forward(self, x, y=None):
        if self.gn is not None:
            return self.gn(x)
        x_size = x.size()
        betas = self.beta_embedding(y).unsqueeze(2).expand(x_size)
        input_batch_major = x.view(x.size(0), -1)
        mean = input_batch_major.mean(dim=1).view(-1, 1, 1).expand(x_size)
        var = input_batch_major.var(dim=1).view(-1, 1, 1).expand(x_size)
        gammas = self.gamma_embedding(y).unsqueeze(2).expand(x_size)
        return gammas * (x - mean) * torch.rsqrt(var + EPSILON) + betas


class EqualizedConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, spectral=False, equalized=True):
        super(EqualizedConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, padding=padding, bias=True)
        self.conv.bias.data.zero_()
        if spectral:
            self.conv = spectral_norm(self.conv)
        if not equalized:
            torch.nn.init.kaiming_normal_(self.conv.weight, a=calculate_gain('conv1d'))
            self.scale = 1.0
        else:
            # torch.nn.init.normal_(self.conv.weight)
            # fan = _calculate_correct_fan(self.conv.weight, 'fan_in')
            # gain = calculate_gain('leaky_relu', 0)
            # std = gain / math.sqrt(fan)
            # self.scale = 1.0 / std
            torch.nn.init.kaiming_normal_(self.conv.weight, a=calculate_gain('conv1d'))
            self.scale = ((torch.mean(self.conv.weight.data ** 2)) ** 0.5).item()
        self.conv.weight.data.copy_(self.conv.weight.data / self.scale)

    def forward(self, x):
        return self.conv(x * self.scale)


class GeneralConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, equalized=True, pad=None, act_alpha=0, do=0,
                 do_mode='mul', num_classes=0, act_norm=None, spectral=False):
        super(GeneralConv, self).__init__()
        pad = (kernel_size - 1) // 2 if pad is None else pad
        conv = EqualizedConv1d(in_channels, out_channels, kernel_size, padding=pad, spectral=spectral,
                               equalized=equalized)
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
