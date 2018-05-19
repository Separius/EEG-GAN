import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from utils import cudize, is_torch4


def pixel_norm(h):
    mean = torch.mean(h * h, dim=1, keepdim=True)
    dom = torch.rsqrt(mean + 1e-8)
    return h * dom


class PGConv1d(nn.Module):
    def __init__(self, ch_in, ch_out, ksize=3, stride=1, pad=1, pixelnorm=True, wscale=True, act='lrelu'):
        super(PGConv1d, self).__init__()
        if wscale:
            if is_torch4:
                init = lambda x: nn.init.kaiming_normal_(x)
            else:
                init = lambda x: nn.init.kaiming_normal(x)
        else:
            init = lambda x: x
        self.conv = nn.Conv1d(ch_in, ch_out, ksize, stride, pad)
        init(self.conv.weight)
        if wscale:
            self.c = np.sqrt(torch.mean(self.conv.weight.data ** 2))
            self.conv.weight.data /= self.c
        else:
            self.c = 1.
        self.pixelnorm = pixelnorm
        if act is not None:
            self.act = nn.LeakyReLU(0.2) if act == 'lrelu' else nn.ReLU()
        else:
            self.act = None
        self.conv = cudize(self.conv)

    def forward(self, x):
        h = x * self.c
        h = self.conv(h)
        if self.act is not None:
            h = self.act(h)
        if self.pixelnorm:
            h = pixel_norm(h)
        return h


class GFirstBlock(nn.Module):
    def __init__(self, ch_in, ch_out, num_channels, **layer_settings):
        super(GFirstBlock, self).__init__()
        self.c1 = PGConv1d(ch_in, ch_out, ksize=4, pad=3, **layer_settings)
        self.c2 = PGConv1d(ch_out, ch_out, **layer_settings)
        self.toRGB = PGConv1d(ch_out, num_channels, ksize=1, pad=0, pixelnorm=False, act=None)

    def forward(self, x, last=False):
        x = self.c1(x)
        x = self.c2(x)
        return self.toRGB(x) if last else x


class GBlock(nn.Module):
    def __init__(self, ch_in, ch_out, num_channels, **layer_settings):
        super(GBlock, self).__init__()
        self.c1 = PGConv1d(ch_in, ch_out, **layer_settings)
        self.c2 = PGConv1d(ch_out, ch_out, **layer_settings)
        self.toRGB = nn.Sequential(PGConv1d(ch_out, num_channels, ksize=1, pad=0, pixelnorm=False, act=None), nn.Tanh())

    def forward(self, x, last=False):
        x = self.c1(x)
        x = self.c2(x)
        return self.toRGB(x) if last else x


class Generator(nn.Module):
    def __init__(self, dataset_shape,  # Overriden based on the dataset ((77, 1, 512, 512))
                 fmap_base=2048, fmap_decay=1.0, fmap_max=256, latent_size=256,
                 normalize_latents=True, wscale=True, pixelnorm=True, leakyrelu=True):
        super(Generator, self).__init__()
        resolution = dataset_shape[-1]
        num_channels = dataset_shape[1]
        print('dataset_shape:', dataset_shape)
        R = int(np.log2(resolution))
        assert resolution == 2 ** R and resolution >= 4

        def nf(stage):
            return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

        if latent_size is None:
            latent_size = nf(0)
        self.normalize_latents = normalize_latents
        layer_settings = {'wscale': wscale, 'pixelnorm': pixelnorm, 'act': 'lrelu' if leakyrelu else 'relu'}
        self.block0 = GFirstBlock(latent_size, nf(1), num_channels, **layer_settings)
        self.blocks = nn.ModuleList([GBlock(nf(i - 1), nf(i), num_channels, **layer_settings) for i in range(2, R)])
        self.depth = 0
        self.alpha = 1.0
        self.latent_size = latent_size
        self.max_depth = len(self.blocks)

    def forward(self, x):  # input: (bs, latent_size)
        h = x.unsqueeze(2)  # (bs, latent_size, 1)
        if self.normalize_latents:
            h = pixel_norm(h)
        h = self.block0(h, self.depth == 0)
        if self.depth > 0:
            for i in range(self.depth - 1):
                h = F.upsample(h, scale_factor=2)
                h = self.blocks[i](h)
            h = F.upsample(h, scale_factor=2)
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
    def __init__(self, ch_in, ch_out, num_channels, **layer_settings):
        super(DLastBlock, self).__init__()
        self.fromRGB = PGConv1d(num_channels, ch_in, ksize=1, pad=0, pixelnorm=False)
        self.stddev = MinibatchStddev()
        self.c1 = PGConv1d(ch_in + 1, ch_in, **layer_settings)
        self.c2 = PGConv1d(ch_in, ch_out, 4, 1, 0, **layer_settings)

    def forward(self, x, first=False):
        if first:
            x = self.fromRGB(x)
        x = self.stddev(x)
        x = self.c1(x)
        x = self.c2(x)
        return x


class MinibatchStddev(nn.Module):
    def __init__(self):
        super(MinibatchStddev, self).__init__()

    def forward(self, x):
        stddev_mean = torch.sqrt(((x - x.mean()) ** 2).mean() + 1.0e-8)
        new_channel = stddev_mean.expand(x.size(0), 1, x.size(2))
        h = torch.cat((x, new_channel), dim=1)
        return h


class Discriminator(nn.Module):
    def __init__(self, dataset_shape,  # Overriden based on dataset
                 fmap_base=2048, fmap_decay=1.0, fmap_max=256,
                 wscale=True, pixelnorm=False, leakyrelu=True):
        super(Discriminator, self).__init__()
        resolution = dataset_shape[-1]
        num_channels = dataset_shape[1]
        R = int(np.log2(resolution))
        assert resolution == 2 ** R and resolution >= 4
        self.R = R

        def nf(stage):
            return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

        layer_settings = {'wscale': wscale, 'pixelnorm': pixelnorm, 'act': 'lrelu' if leakyrelu else 'relu'}
        self.blocks = nn.ModuleList(
            [DBlock(nf(i), nf(i - 1), num_channels, **layer_settings) for i in range(R - 1, 1, -1)] + [
                DLastBlock(nf(1), nf(0), num_channels, **layer_settings)])

        self.linear = nn.Linear(nf(0), 1)
        self.depth = 0
        self.alpha = 1.0
        self.max_depth = len(self.blocks) - 1

    def forward(self, x):
        xhighres = x
        h = self.blocks[-(self.depth + 1)](xhighres, True)
        if self.depth > 0:
            h = F.avg_pool1d(h, 2)
            if self.alpha < 1.0:
                xlowres = F.avg_pool1d(xhighres, 2)
                preult_rgb = self.blocks[-self.depth].fromRGB(xlowres)
                h = h * self.alpha + (1 - self.alpha) * preult_rgb
        for i in range(self.depth, 0, -1):
            h = self.blocks[-i](h)
            if i > 1:
                h = F.avg_pool1d(h, 2)
        h = self.linear(h.squeeze(-1))
        return h
