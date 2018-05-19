import numpy as np
import torch
from torch import nn
from utils import cudize, is_torch4
from torch.nn.init import kaiming_normal, calculate_gain


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
    def __init__(self, c_in, c_out, k_size, stride=1, padding=0):
        super(EqualizedConv1d, self).__init__()
        self.conv = nn.Conv1d(c_in, c_out, k_size, stride, padding, bias=False)
        if is_torch4:
            torch.nn.init.kaiming_normal_(self.conv.weight, a=calculate_gain('conv1d'))
        else:
            kaiming_normal(self.conv.weight, a=calculate_gain('conv1d'))
        self.bias = torch.nn.Parameter(torch.FloatTensor(c_out).fill_(0))
        self.scale = (torch.mean(self.conv.weight.data ** 2)) ** 0.5
        self.conv.weight.data.copy_(self.conv.weight.data / self.scale)
        self.scale = cudize(self.scale)

    def forward(self, x):
        x = self.conv(x.mul(self.scale))
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

        rnd_shape = [s if axis in self.axes else 1 for axis, s in
                     enumerate(x.size())]  # [x.size(axis) for axis in self.axes]
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
        rnd = torch.autograd.Variable(torch.from_numpy(rnd).type(x.data.type()))
        if x.is_cuda:
            rnd = rnd.cuda()
        return x * rnd

    def __repr__(self):
        param_str = '(mode = %s, strength = %s, axes = %s, normalize = %s)' % (
            self.mode, self.strength, self.axes, self.normalize)
        return self.__class__.__name__ + param_str


class PGConv1d(nn.Module):
    def __init__(self, ch_in, ch_out, ksize=3, stride=1, pad=1, pixelnorm=True, act='lrelu', bn=False, do=0):
        super(PGConv1d, self).__init__()
        self.net = [EqualizedConv1d(ch_in, ch_out, ksize, stride, pad)]
        if bn:
            self.net.append(nn.BatchNorm1d(num_features=ch_out))
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
            self.net.append(GDropLayer(strength=do))
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        return self.net(x)


class GFirstBlock(nn.Module):
    def __init__(self, ch_in, ch_out, num_channels, **layer_settings):
        super(GFirstBlock, self).__init__()
        self.c1 = PGConv1d(ch_in, ch_out, ksize=4, pad=3, **layer_settings)
        self.c2 = PGConv1d(ch_out, ch_out, **layer_settings)
        self.toRGB = nn.Sequential(PGConv1d(ch_out, num_channels, ksize=1, pad=0, pixelnorm=False, act=None),
                                   ScaledTanh())

    def forward(self, x, last=False):
        x = self.c1(x)
        x = self.c2(x)
        return self.toRGB(x) if last else x


class GBlock(nn.Module):
    def __init__(self, ch_in, ch_out, num_channels, **layer_settings):
        super(GBlock, self).__init__()
        self.c1 = PGConv1d(ch_in, ch_out, **layer_settings)
        self.c2 = PGConv1d(ch_out, ch_out, **layer_settings)
        self.toRGB = nn.Sequential(PGConv1d(ch_out, num_channels, ksize=1, pad=0, pixelnorm=False, act=None),
                                   ScaledTanh())

    def forward(self, x, last=False):
        x = self.c1(x)
        x = self.c2(x)
        return self.toRGB(x) if last else x


class Generator(nn.Module):
    def __init__(self, dataset_shape, fmap_base=2048, fmap_max=256, latent_size=256, upsample='nearest',
                 normalize_latents=True, pixelnorm=True, activation='lrelu', dropout=0, batchnorm=False):
        super(Generator, self).__init__()
        resolution = dataset_shape[-1]
        num_channels = dataset_shape[1]
        R = int(np.log2(resolution))
        assert resolution == 2 ** R and resolution >= 4

        def nf(stage):
            return min(int(fmap_base / (2.0 ** stage)), fmap_max)

        if latent_size is None:
            latent_size = nf(0)
        self.normalize_latents = normalize_latents
        layer_settings = {'pixelnorm': pixelnorm, 'act': activation, 'do': dropout, 'bn': batchnorm}
        self.block0 = GFirstBlock(latent_size, nf(1), num_channels, **layer_settings)
        self.blocks = nn.ModuleList([GBlock(nf(i - 1), nf(i), num_channels, **layer_settings) for i in range(2, R)])
        self.depth = 0
        self.alpha = 1.0
        self.latent_size = latent_size
        self.max_depth = len(self.blocks)
        self.upsampler = nn.Upsample(scale_factor=2, mode=upsample)

    def forward(self, x):  # input: (bs, latent_size)
        h = x.unsqueeze(2)  # (bs, latent_size, 1)
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
    def __init__(self, ch_in, ch_out, num_channels, **layer_settings):
        super(DBlock, self).__init__()
        self.fromRGB = PGConv1d(num_channels, ch_in, ksize=1, pad=0, pixelnorm=False)
        self.c1 = PGConv1d(ch_in, ch_in, **layer_settings)
        self.c2 = PGConv1d(ch_in, ch_out, **layer_settings)

    def forward(self, x, first=False):
        if first:
            x = self.fromRGB(x)
        x = self.c1(x)
        x = self.c2(x)
        return x


class DLastBlock(nn.Module):
    def __init__(self, ch_in, ch_out, num_channels, apply_sigmoid=False, **layer_settings):
        super(DLastBlock, self).__init__()
        self.fromRGB = PGConv1d(num_channels, ch_in, ksize=1, pad=0, pixelnorm=False)
        self.net = [MinibatchStddev(), PGConv1d(ch_in + 1, ch_in, **layer_settings),
                    PGConv1d(ch_in, ch_out, 4, 1, 0, **layer_settings)]
        if apply_sigmoid:
            self.net.append(nn.Sigmoid())
        self.net = nn.Sequential(*self.net)

    def forward(self, x, first=False):
        if first:
            x = self.fromRGB(x)
        return self.net(x)


class MinibatchStddev(nn.Module):
    def __init__(self):
        super(MinibatchStddev, self).__init__()

    def forward(self, x):
        stddev_mean = torch.sqrt(((x - x.mean()) ** 2).mean() + 1.0e-8)
        new_channel = stddev_mean.expand(x.size(0), 1, x.size(2))
        h = torch.cat((x, new_channel), dim=1)
        return h


class Discriminator(nn.Module):
    def __init__(self, dataset_shape, fmap_base=2048, fmap_decay=1.0, fmap_max=256, downsample='average',
                 apply_sigmoid=False, pixelnorm=False, activation='lrelu', dropout=0, batchnorm=False):
        super(Discriminator, self).__init__()
        resolution = dataset_shape[-1]
        num_channels = dataset_shape[1]
        R = int(np.log2(resolution))
        assert resolution == 2 ** R and resolution >= 4
        self.R = R

        def nf(stage):
            return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

        layer_settings = {'pixelnorm': pixelnorm, 'act': activation, 'do': dropout, 'bn': batchnorm}
        self.blocks = nn.ModuleList(
            [DBlock(nf(i), nf(i - 1), num_channels, **layer_settings) for i in range(R - 1, 1, -1)] + [
                DLastBlock(nf(1), nf(0), num_channels, apply_sigmoid, **layer_settings)])

        self.linear = nn.Linear(nf(0), 1)
        self.depth = 0
        self.alpha = 1.0
        self.max_depth = len(self.blocks) - 1
        if downsample == 'average':
            self.downsampler = nn.AvgPool1d(kernel_size=2)
        else:
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
        h = self.linear(h.squeeze(-1))
        return h
