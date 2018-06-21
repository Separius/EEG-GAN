import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from utils import cudize, pixel_norm
from torch.nn.init import calculate_gain
from spectral_norm import spectral_norm as spectral_norm_wrapper
from torch.nn.utils import weight_norm as weight_norm_wrapper


class PhaseShuffle(nn.Module):
    """
    Performs phase shuffling, i.e. shifting feature axis of a 3D tensor
    by a random integer in {-n, n} and performing reflection padding where
    necessary
    If batch shuffle is enabled, only a single shuffle is applied to the entire
    batch, rather than each sample in the batch.
    """

    def __init__(self, shift_factor, batch_shuffle=False):
        super(PhaseShuffle, self).__init__()
        self.shift_factor = shift_factor
        self.batch_shuffle = batch_shuffle

    def forward(self, x):
        if self.shift_factor == 0:
            return x

        if self.batch_shuffle:
            k = int(torch.Tensor(1).random_(0, 2 * self.shift_factor + 1)) - self.shift_factor
            if k == 0:
                return x
            if k > 0:
                x_trunc = x[:, :, :-k]
                pad = (k, 0)
            else:
                x_trunc = x[:, :, -k:]
                pad = (0, -k)
            x_shuffle = F.pad(x_trunc, pad, mode='reflect')
        else:
            k_list = torch.Tensor(x.shape[0]).random_(0, 2 * self.shift_factor + 1) - self.shift_factor
            k_list = k_list.numpy().astype(int)
            k_map = {}
            for idx, k in enumerate(k_list):
                k = int(k)
                if k not in k_map:
                    k_map[k] = []
                k_map[k].append(idx)
            x_shuffle = x.clone()
            for k, idxs in k_map.items():
                if k > 0:
                    x_shuffle[idxs] = F.pad(x[idxs][..., :-k], (k, 0), mode='reflect')
                else:
                    x_shuffle[idxs] = F.pad(x[idxs][..., -k:], (0, -k), mode='reflect')
        assert x_shuffle.shape == x.shape, "{}, {}".format(x_shuffle.shape, x.shape)
        return x_shuffle


class DownSample(nn.Module):
    def __init__(self, scale_factor=2):
        super(DownSample, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, input):
        return input[:, :, ::self.scale_factor]


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


class ChannelByChannelOut(nn.Module):
    def __init__(self, ch_in, ch_out, **layer_settings):
        super(ChannelByChannelOut, self).__init__()
        self.num_out = ch_out
        self.out = nn.ModuleList(
            [nn.Sequential(GeneralConv(ch_in + i, 1, **layer_settings), ScaledTanh()) for i in range(ch_out)])

    def forward(self, x):
        r = x
        for o in self.out:
            r = torch.cat([r, o(r)], dim=1)
        return r[:, -self.num_out:, :]


class SelfAttention(nn.Module):
    def __init__(self, channels_in, key_length):
        super(SelfAttention, self).__init__()
        self.gamma = 0
        self.channels_in = channels_in
        self.key_length = key_length
        self.to_key = nn.Conv1d(channels_in, key_length, kernel_size=1)
        self.to_query = nn.Conv1d(channels_in, key_length, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, v):
        if self.gamma == 0:
            return v
        T = v.size(2)
        k = self.to_key(v)  # k, q = (N, C, T)
        q = self.to_query(v)
        e1 = q.unsqueeze(3).repeat(1, 1, 1, T).permute(0, 1, 3, 2)
        e2 = k.unsqueeze(3).repeat(1, 1, 1, T)
        a = self.softmax((e1 * e2).sum(dim=1))  # a is (N, T(normalized), T)
        a = torch.bmm(v, a)
        return v + self.gamma * a


def get_activation(act, ch_out):
    if act == 'prelu':
        return nn.PReLU(num_parameters=ch_out)
    elif act == 'lrelu':
        return nn.LeakyReLU(0.2, inplace=True)
    elif act == 'relu':
        return nn.ReLU(inplace=True)
    elif act == 'relu6':
        return nn.ReLU6(inplace=True)
    elif act == 'elu':
        return nn.ELU(inplace=True)
    raise ValueError()


class Concatenate(nn.Module):
    def __init__(self, module_list):
        super(Concatenate, self).__init__()
        self.module_list = module_list

    def forward(self, *args):
        return torch.cat([m(*args) for m in self.module_list], dim=1)


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
    def __init__(self, c_in, c_out, k_size, padding=0, is_spectral=False, is_weight_norm=False, groups=1):
        super(EqualizedConv1d, self).__init__()
        self.conv = nn.Conv1d(c_in, c_out, k_size, padding=padding, bias=False, groups=groups)
        if is_spectral:
            self.conv = spectral_norm_wrapper(self.conv)
        if is_weight_norm:
            self.conv = weight_norm_wrapper(self.conv)
        torch.nn.init.kaiming_normal_(self.conv.weight, a=calculate_gain('conv1d'))
        self.bias = torch.nn.Parameter(torch.FloatTensor(c_out).zero_())
        self.scale = ((torch.mean(self.conv.weight.data ** 2)) ** 0.5).item()
        self.conv.weight.data.copy_(self.conv.weight.data / self.scale)

    def forward(self, x):
        x = self.conv(x * self.scale)
        return x + self.bias.view(1, -1, 1).expand_as(x)


class ToRGB(nn.Module):
    def __init__(self, ch_in, num_channels, normalization=None, ch_by_ch=False, equalized=True):
        super(ToRGB, self).__init__()
        self.num_channels = num_channels
        layer_settings = dict(ksize=1, pixelnorm=False, act=None, equalized=equalized, normalization=normalization,
                              spreading_factor=0)
        if ch_by_ch:
            self.toRGB = ChannelByChannelOut(ch_in, num_channels, **layer_settings)
        else:
            self.toRGB = nn.Sequential(GeneralConv(ch_in, num_channels, **layer_settings), ScaledTanh())

    def forward(self, x):
        return self.toRGB(x)


class NeoPGConv1d(nn.Module):
    def __init__(self, ch_in, ch_out, ksize=3, equalized=True, pad=None, pixelnorm=True, act='lrelu', do=0,
                 do_mode='mul', spectral=False, phase_shuffle=0, normalization=None, num_classes=0, groups=1):
        super(NeoPGConv1d, self).__init__()
        pad = (ksize - 1) // 2 if pad is None else pad
        if equalized:
            conv = EqualizedConv1d(ch_in, ch_out, ksize, padding=pad, is_spectral=spectral,
                                   is_weight_norm=normalization == 'weight_norm', groups=groups)
        else:
            conv = nn.Conv1d(ch_in, ch_out, ksize, padding=pad, groups=groups)
            if spectral:
                conv = spectral_norm_wrapper(conv)
            if normalization == 'weight_norm':
                conv = weight_norm_wrapper(conv)
        norm = None
        if normalization:
            if normalization == 'layer_norm':
                norm = ConditionalGroupNorm(ch_out, num_classes)
            elif normalization == 'batch_norm':
                norm = ConditionalBatchNorm(ch_out, num_classes)
        self.conv = conv
        self.norm = norm
        self.net = []
        if act:
            self.net.append(get_activation(act, ch_out))
        if pixelnorm:
            self.net.append(PixelNorm())
        if do != 0:
            self.net.append(GDropLayer(strength=do, mode=do_mode))
        if phase_shuffle != 0:
            self.net.append(PhaseShuffle(phase_shuffle))
        self.net = nn.Sequential(*self.net)

    def forward(self, x, y=None):
        if self.norm:
            return self.net(self.norm(self.conv(x), y))
        return self.net(self.conv(x))


class GeneralConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, spreading_factor=0, equalized=True, pad=None,
                 pixelnorm=True, act='lrelu', do=0, do_mode='mul', spectral=False, phase_shuffle=0, normalization=None,
                 num_classes=0):
        super(GeneralConv, self).__init__()
        self.is_separable = (spreading_factor != 0)
        if self.is_separable:
            self.c1 = NeoPGConv1d(in_channels, in_channels * spreading_factor, ksize, equalized=equalized,
                                  phase_shuffle=phase_shuffle, pixelnorm=pixelnorm,
                                  act=act if act is not None else 'relu', do=do, do_mode=do_mode, spectral=spectral,
                                  pad=pad, normalization=normalization, num_classes=num_classes, groups=in_channels)
            self.c2 = NeoPGConv1d(in_channels * spreading_factor, out_channels, ksize=1, equalized=equalized,
                                  pixelnorm=pixelnorm, act=act, do=do, do_mode=do_mode, spectral=spectral,
                                  normalization=normalization, num_classes=num_classes)
        else:
            self.c1 = NeoPGConv1d(in_channels, out_channels, ksize, equalized=equalized, pad=pad,
                                  pixelnorm=pixelnorm, act=act, do=do, do_mode=do_mode, spectral=spectral,
                                  phase_shuffle=phase_shuffle, normalization=normalization, num_classes=num_classes)

    def forward(self, x, y=None):
        x = self.c1(x, y)
        if self.is_separable:
            return self.c2(x, y)
        return x
