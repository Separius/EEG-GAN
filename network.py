import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from utils import cudize
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


def pixel_norm(h):
    mean = torch.mean(h * h, dim=1, keepdim=True)
    dom = torch.rsqrt(mean + 1e-8)
    return h * dom


class DownSample(nn.Module):
    def __init__(self, scale_factor=1):
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


class EqualizedConv1d(nn.Module):
    def __init__(self, c_in, c_out, k_size, stride=1, padding=0, is_spectral=False, is_weight_norm=False):
        super(EqualizedConv1d, self).__init__()
        self.conv = nn.Conv1d(c_in, c_out, k_size, stride, padding, bias=False)
        if is_spectral:
            self.conv = spectral_norm_wrapper(self.conv)
        if is_weight_norm:
            self.conv = weight_norm_wrapper(self.conv)
        torch.nn.init.kaiming_normal_(self.conv.weight, a=calculate_gain('conv1d'))
        self.bias = torch.nn.Parameter(torch.FloatTensor(c_out).fill_(0))
        self.scale = ((torch.mean(self.conv.weight.data ** 2)) ** 0.5).item()
        self.conv.weight.data.copy_(self.conv.weight.data / self.scale)

    def forward(self, x):
        x = self.conv(x * self.scale)
        return x + self.bias.view(1, -1, 1).expand_as(x)


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
    def __init__(self, ch_in, ch_out, normalization=None, equalized=True):
        super(ChannelByChannelOut, self).__init__()
        self.num_out = ch_out
        self.out = nn.ModuleList([nn.Sequential(
            NeoPGConv1d(ch_in + i, 1, ksize=1, pixelnorm=False, act=None, equalized=equalized,
                        normalization=normalization), ScaledTanh()) for i in range(ch_out)])

    def forward(self, x):
        r = x
        for o in self.out:
            r = torch.cat([r, o(r)], dim=1)
        return r[:, -self.num_out:, :]


class ToRGB(nn.Module):
    def __init__(self, ch_in, num_channels, normalization=None, ch_by_ch=False, equalized=True):
        super(ToRGB, self).__init__()
        self.num_channels = num_channels
        if ch_by_ch:
            self.toRGB = ChannelByChannelOut(ch_in, num_channels, normalization=normalization, equalized=equalized)
        else:
            self.toRGB = nn.Sequential(
                NeoPGConv1d(ch_in, num_channels, ksize=1, pixelnorm=False, act=None, equalized=equalized,
                            normalization=normalization), ScaledTanh())

    def forward(self, x):
        return self.toRGB(x)


class NeoPGConv1d(nn.Module):
    def __init__(self, ch_in, ch_out, ksize=3, equalized=True, pad=None, pixelnorm=True,
                 act='lrelu', do=0, do_mode='mul', spectral=False, phase_shuffle=0, normalization=None):
        super(NeoPGConv1d, self).__init__()
        pad = (ksize - 1) // 2 if pad is None else pad
        if equalized:
            conv = EqualizedConv1d(ch_in, ch_out, ksize, 1, pad, is_spectral=spectral,
                                   is_weight_norm=normalization == 'weight_norm')
        else:
            conv = nn.Conv1d(ch_in, ch_out, ksize, 1, pad)
            if spectral:
                conv = spectral_norm_wrapper(conv)
            if normalization == 'weight_norm':
                conv = weight_norm_wrapper(conv)
        norm = None
        if normalization:
            if normalization == 'layer_norm':
                norm = nn.LayerNorm(ch_out)
            elif normalization == 'batch_norm':
                norm = nn.BatchNorm1d(ch_out)
        self.net = [conv]
        if norm is not None:
            self.net.append(norm)
        if act:
            if act == 'prelu':
                self.net.append(nn.PReLU(num_parameters=ch_out))
            elif act == 'lrelu':
                self.net.append(nn.LeakyReLU(0.2, inplace=True))
            elif act == 'relu':
                self.net.append(nn.ReLU(inplace=True))
            elif act == 'relu6':
                self.net.append(nn.ReLU6(inplace=True))
            elif act == 'elu':
                self.net.append(nn.ELU(inplace=True))
        if pixelnorm:
            self.net.append(PixelNorm())
        if do != 0:
            self.net.append(GDropLayer(strength=do, mode=do_mode))
        if phase_shuffle != 0:
            self.net.append(PhaseShuffle(phase_shuffle))
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        return self.net(x)


class GBlock(nn.Module):
    def __init__(self, ch_in, ch_out, num_channels, ksize=3, equalized=True, initial_size=None, ch_by_ch=False,
                 normalization=None, residual=False, **layer_settings):
        super(GBlock, self).__init__()
        is_first = initial_size is not None
        c1 = NeoPGConv1d(ch_in, ch_out, equalized=equalized, ksize=2 ** initial_size if is_first else ksize,
                         pad=2 ** initial_size - 1 if is_first else None, normalization=normalization,
                         **layer_settings)
        c2 = NeoPGConv1d(ch_out, ch_out, equalized=equalized, normalization=normalization, **layer_settings)
        self.bypass = nn.Sequential() if not residual or is_first else NeoPGConv1d(ch_in, ch_out, ksize=1,
                                                                                   equalized=equalized,
                                                                                   pixelnorm=False, act=None,
                                                                                   spectral=layer_settings[
                                                                                       'spectral_norm'])
        self.residual = nn.Sequential(c1, c2)
        self.toRGB = ToRGB(ch_out, num_channels, normalization=None if normalization == 'batch_norm' else normalization,
                           ch_by_ch=ch_by_ch, equalized=equalized)

    def forward(self, x, last=False):
        x = self.residual(x) + self.bypass(x)
        return self.toRGB(x) if last else x


class Generator(nn.Module):
    def __init__(self, dataset_shape, initial_size, fmap_base=2048, fmap_max=256, fmap_min=16, latent_size=256,
                 upsample='linear', normalize_latents=True, pixelnorm=True, activation='lrelu', dropout=0.1,
                 residual=False, do_mode='mul', equalized=True, spectral_norm=False, ch_by_ch=False, kernel_size=3,
                 normalization=None):
        super(Generator, self).__init__()
        resolution = dataset_shape[-1]
        num_channels = dataset_shape[1]
        R = int(np.log2(resolution))
        assert resolution == 2 ** R and resolution >= 2 ** initial_size

        def nf(stage):
            return min(max(int(fmap_base / (2.0 ** stage)), fmap_min), fmap_max)

        if latent_size is None:
            latent_size = nf(initial_size - 2)
        self.normalize_latents = normalize_latents
        layer_settings = dict(pixelnorm=pixelnorm, act=activation, do=dropout, do_mode=do_mode, spectral=spectral_norm)
        self.block0 = GBlock(latent_size, nf(1), num_channels, ksize=kernel_size, equalized=equalized,
                             initial_size=initial_size, ch_by_ch=ch_by_ch, normalization=normalization,
                             residual=residual, **layer_settings)
        self.blocks = nn.ModuleList([GBlock(nf(i - 1), nf(i), num_channels, ksize=kernel_size, equalized=equalized,
                                            ch_by_ch=ch_by_ch, normalization=normalization, residual=residual,
                                            **layer_settings) for i in range(initial_size, R)])
        self.depth = 0
        self.alpha = 1.0
        self.latent_size = latent_size
        self.max_depth = len(self.blocks)
        if upsample == 'linear':
            self.upsampler = nn.Upsample(scale_factor=2, mode=upsample, align_corners=True)
        else:
            self.upsampler = nn.Upsample(scale_factor=2, mode=upsample)

    def forward(self, x):
        h = x.unsqueeze(2)
        if self.normalize_latents:
            h = pixel_norm(h)
        h = self.block0(h, self.depth == 0)
        if self.depth > 0:
            for i in range(self.depth - 1):
                h = self.upsampler(h)
                h = self.blocks[i](h)
            h = self.upsampler(h)
            ult = self.blocks[self.depth - 1](h, True)
            if self.alpha < 1.0:
                if self.depth > 1:
                    preult_rgb = self.blocks[self.depth - 2].toRGB(h)
                else:
                    preult_rgb = self.block0.toRGB(h)
            else:
                preult_rgb = 0
            h = preult_rgb * (1 - self.alpha) + ult * self.alpha
        return h


class DBlock(nn.Module):
    def __init__(self, ch_in, ch_out, num_channels, initial_size=None, temporal=False, num_stat_channels=1,
                 ksize=3, equalized=True, spectral=False, normalization=None, residual=False, **layer_settings):
        super(DBlock, self).__init__()
        is_last = initial_size is not None
        self.bypass = nn.Sequential() if not residual or is_last else NeoPGConv1d(ch_in, ch_out, ksize=1,
                                                                                  equalized=equalized, pixelnorm=False,
                                                                                  act=None, spectral=spectral)
        self.fromRGB = NeoPGConv1d(num_channels, ch_in, ksize=1, pixelnorm=False, equalized=equalized,
                                   spectral=spectral,
                                   normalization=None if normalization == 'batch_norm' else normalization)
        if num_stat_channels > 0 and is_last:
            self.net = [MinibatchStddev(temporal, num_stat_channels)]
        else:
            self.net = []
        self.net.append(
            NeoPGConv1d(ch_in + (num_stat_channels if is_last else 0), ch_in, ksize=ksize, equalized=equalized,
                        spectral=spectral, normalization=normalization, **layer_settings))
        self.net.append(
            NeoPGConv1d(ch_in, ch_out, ksize=(2 ** initial_size) if is_last else ksize, pad=0 if is_last else None,
                        equalized=equalized, spectral=spectral, normalization=normalization, **layer_settings))
        self.net = nn.Sequential(*self.net)

    def forward(self, x, first=False):
        if first:
            x = self.fromRGB(x)
        return self.net(x) + self.bypass(x)


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


class Discriminator(nn.Module):
    def __init__(self, dataset_shape, initial_size, fmap_base=2048, fmap_max=256, fmap_min=64, downsample='average',
                 pixelnorm=False, activation='lrelu', dropout=0.1, do_mode='mul', equalized=True, spectral_norm=False,
                 kernel_size=3, phase_shuffle=0, temporal_stats=False, num_stat_channels=1, normalization=None,
                 residual=False):
        super(Discriminator, self).__init__()
        resolution = dataset_shape[-1]
        num_channels = dataset_shape[1]
        R = int(np.log2(resolution))
        assert resolution == 2 ** R and resolution >= initial_size ** 2
        self.R = R

        def nf(stage):
            return max(min(max(int(fmap_base / (2.0 ** stage)), fmap_min), fmap_max), 2)

        layer_settings = dict(pixelnorm=pixelnorm, act=activation, do=dropout, do_mode=do_mode,
                              phase_shuffle=phase_shuffle)
        last_block = DBlock(nf(initial_size - 1), nf(initial_size - 2), num_channels, initial_size=initial_size,
                            temporal=temporal_stats, num_stat_channels=num_stat_channels, ksize=kernel_size,
                            residual=residual, equalized=equalized, spectral=spectral_norm, normalization=normalization,
                            **layer_settings)
        self.blocks = nn.ModuleList([DBlock(nf(i), nf(i - 1), num_channels, ksize=kernel_size, equalized=equalized,
                                            initial_size=None, residual=residual, spectral=spectral_norm,
                                            normalization=normalization, **layer_settings) for i in
                                     range(R - 1, initial_size - 1, -1)] + [last_block])
        self.linear = nn.Linear(nf(initial_size - 2), 1)
        if spectral_norm:
            self.linear = spectral_norm_wrapper(self.linear)
        self.depth = 0
        self.alpha = 1.0
        self.max_depth = len(self.blocks) - 1
        if downsample == 'average':
            self.downsampler = nn.AvgPool1d(kernel_size=2)
        elif downsample == 'stride':
            self.downsampler = DownSample(scale_factor=2)

    def forward(self, x):
        xhighres = x
        h = self.blocks[-(self.depth + 1)](xhighres, True)
        if self.depth > 0:
            h = self.downsampler(h)
            if self.alpha < 1.0:
                xlowres = self.downsampler(xhighres)
                preult_rgb = self.blocks[-self.depth].fromRGB(xlowres)
                h = h * self.alpha + (1 - self.alpha) * preult_rgb
        for i in range(self.depth, 0, -1):
            h = self.blocks[-i](h)
            if i > 1:
                h = self.downsampler(h)
        h = h.squeeze(-1)
        return self.linear(h), h
