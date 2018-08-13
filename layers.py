import math
import torch
import numpy as np
from torch import nn
from utils import cudize, pixel_norm
from torch.nn.init import calculate_gain
from torch.nn.utils import weight_norm, spectral_norm


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
    def __init__(self, channels_in):
        super(SelfAttention, self).__init__()
        self.gamma = 0
        self.channels_in = channels_in
        d_key = max(channels_in // 8, 2)
        self.to_key = nn.Conv1d(channels_in, d_key, kernel_size=1)
        self.to_query = nn.Conv1d(channels_in, d_key, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)
        self.scale = math.sqrt(d_key)

    def forward(self, v):
        if self.gamma == 0:
            return v
        T = v.size(2)
        k = self.to_key(v)  # k, q = (N, C, T)
        q = self.to_query(v)
        e1 = q.unsqueeze(3).repeat(1, 1, 1, T).permute(0, 1, 3, 2)
        e2 = k.unsqueeze(3).repeat(1, 1, 1, T)
        a = self.softmax((e1 * e2).sum(dim=1) / self.scale)  # a is (N, T(normalized), T)
        a = torch.bmm(v, a)
        return v + self.gamma * a

    def __repr__(self):
        param_str = '(channels_in = {})'.format(self.channels_in)
        return self.__class__.__name__ + param_str


class MinibatchStddev(nn.Module):
    def __init__(self, temporal=False, out_channels=1):
        super(MinibatchStddev, self).__init__()
        self.temporal = temporal
        self.out_channels = out_channels

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

    def forward(self, x, y=None):
        if y is None:
            return self.bn(x)
        x_size = x.size()
        channels = x_size[1]
        gammas = self.gamma_embedding(y).unsqueeze(2).expand(x_size)
        betas = self.beta_embedding(y).unsqueeze(2).expand(x_size)
        input_channel_major = x.permute(1, 0, 2).contiguous().view(channels, -1)
        mean = input_channel_major.mean(dim=1)
        var = input_channel_major.var(dim=1)
        x = (x - mean.view(1, channels, 1).expand(x_size)) * torch.rsqrt(var.view(1, channels, 1).expand(x_size) + 1e-8)
        return gammas * x + betas


class ConditionalGroupNorm(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(ConditionalGroupNorm, self).__init__()
        if num_classes == 0:
            self.gn = nn.GroupNorm(1, num_channels)
        else:
            self.gamma_embedding = nn.EmbeddingBag(num_classes, num_channels)
            self.gamma_embedding.weight.data.fill_(1.0)
            self.beta_embedding = nn.EmbeddingBag(num_classes, num_channels)
            self.beta_embedding.weight.data.zero_()

    def forward(self, x, y=None):
        if y is None:
            return self.gn(x)
        x_size = x.size()
        gammas = self.gamma_embedding(y).unsqueeze(2).expand(x_size)
        betas = self.beta_embedding(y).unsqueeze(2).expand(x_size)
        input_batch_major = x.view(x.size(0), -1)
        mean = input_batch_major.mean(dim=1).view(-1, 1, 1).expand(x_size)
        var = input_batch_major.var(dim=1).view(-1, 1, 1).expand(x_size)
        return gammas * (x - mean) * torch.rsqrt(var + 1e-8) + betas


class EqualizedConv1d(nn.Module):
    def __init__(self, c_in, c_out, k_size, padding=0, param_norm=None, equalized=True):
        super(EqualizedConv1d, self).__init__()
        self.conv = nn.Conv1d(c_in, c_out, k_size, padding=padding, bias=False)
        if param_norm == 'spectral':
            self.conv = spectral_norm(self.conv)
        elif param_norm == 'weight':
            self.conv = weight_norm(self.conv)
        torch.nn.init.kaiming_normal_(self.conv.weight, a=calculate_gain('conv1d'))
        self.bias = torch.nn.Parameter(torch.FloatTensor(c_out).zero_())
        if equalized:
            self.scale = ((torch.mean(self.conv.weight.data ** 2)) ** 0.5).item()
        else:
            self.scale = 1.0
        self.conv.weight.data.copy_(self.conv.weight.data / self.scale)

    def forward(self, x):
        x = self.conv(x * self.scale)
        return x + self.bias.view(1, -1, 1).expand_as(x)


class GeneralConv(nn.Module):
    def __init__(self, ch_in, ch_out, ksize=3, equalized=True, pad=None, act=True, do=0, do_mode='mul', num_classes=0,
                 act_norm=None, param_norm=None):
        super(GeneralConv, self).__init__()
        pad = (ksize - 1) // 2 if pad is None else pad
        conv = EqualizedConv1d(ch_in, ch_out, ksize, padding=pad, param_norm=param_norm, equalized=equalized)
        norm = None
        if act_norm == 'layer':
            norm = ConditionalGroupNorm(ch_out, num_classes)
        elif act_norm == 'batch':
            norm = ConditionalBatchNorm(ch_out, num_classes)
        self.conv = conv
        self.norm = norm
        self.net = []
        if act:
            self.net.append(nn.ReLU(inplace=True))
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
