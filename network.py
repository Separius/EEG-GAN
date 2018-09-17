import math
from torch import nn
from utils import pixel_norm
from functools import partial
import torch.nn.functional as F
from layers import GeneralConv, SelfAttention, MinibatchStddev, ScaledTanh, spectral_norm


class GBlock(nn.Module):
    def __init__(self, ch_in, ch_out, num_channels, ksize=3, equalized=True, initial_kernel_size=None,
                 is_residual=False, no_tanh=False, spectral=False, sngan_rgb=False, **layer_settings):
        super(GBlock, self).__init__()
        is_first = initial_kernel_size is not None
        self.c1 = GeneralConv(ch_in, ch_out, equalized=equalized,
                              kernel_size=initial_kernel_size if is_first else ksize,
                              pad=initial_kernel_size - 1 if is_first else None, spectral=spectral, **layer_settings)
        self.c2 = GeneralConv(ch_out, ch_out, equalized=equalized, kernel_size=ksize, spectral=spectral,
                              **layer_settings)
        if sngan_rgb:
            to_rgb = nn.Sequential(nn.BatchNorm1d(ch_out), nn.LeakyReLU(0.2, True),
                                   GeneralConv(ch_out, num_channels, kernel_size=3, act_alpha=-1, equalized=equalized,
                                               spectral=spectral))
        else:
            to_rgb = GeneralConv(ch_out, num_channels, kernel_size=1, act_alpha=-1, equalized=equalized,
                                 spectral=spectral)
        if no_tanh:
            self.toRGB = to_rgb
        else:
            self.toRGB = nn.Sequential(to_rgb, ScaledTanh())
        if not is_first and is_residual:
            self.residual = nn.Sequential() if ch_in == ch_out else GeneralConv(ch_in, ch_out, 1, equalized,
                                                                                act_alpha=-1, spectral=spectral)
        else:
            self.residual = None

    def forward(self, x, y=None, last=False):
        h = self.c2(self.c1(x, y), y)
        if last:
            return self.toRGB(h)
        else:
            if self.residual is not None:
                h = h + self.residual(x)
            return h


class Generator(nn.Module):
    def __init__(self, dataset_shape, initial_size, fmap_base, fmap_max, fmap_min, kernel_size, equalized,
                 self_attention_layers, num_classes, sngan_rgb=False, act_alpha=0, latent_size=256, residual=False,
                 normalize_latents=True, dropout=0.1, do_mode='mul', spectral=False, act_norm='pixel', no_tanh=False):
        super(Generator, self).__init__()
        resolution = dataset_shape[-1]
        num_channels = dataset_shape[1]
        progression_scale = 2
        R = int(math.log(resolution, progression_scale))
        assert resolution == progression_scale ** R and resolution >= progression_scale ** initial_size

        def nf(stage):
            return min(max(int(fmap_base / (2.0 ** stage)), fmap_min), fmap_max)

        if latent_size is None:
            latent_size = nf(0)
        self.normalize_latents = normalize_latents
        layer_settings = dict(do=dropout, do_mode=do_mode, num_classes=num_classes,
                              act_norm=act_norm, act_alpha=act_alpha)
        initial_kernel_size = progression_scale ** initial_size
        self.block0 = GBlock(latent_size, nf(1), num_channels, ksize=kernel_size, equalized=equalized,
                             initial_kernel_size=initial_kernel_size, is_residual=residual, spectral=spectral,
                             no_tanh=no_tanh, sngan_rgb=sngan_rgb, **layer_settings)
        dummy = []
        self.self_attention = dict()
        for layer in self_attention_layers:
            dummy.append(SelfAttention(nf(layer + 1), spectral))
            self.self_attention[layer] = dummy[-1]
        self.dummy = nn.ModuleList(dummy)
        self.blocks = nn.ModuleList([GBlock(nf(i - initial_size + 1), nf(i - initial_size + 2), num_channels,
                                            ksize=kernel_size, equalized=equalized, is_residual=residual,
                                            spectral=spectral, no_tanh=no_tanh, sngan_rgb=sngan_rgb, **layer_settings)
                                     for i in range(initial_size, R)])
        self.depth = 0
        self.alpha = 1.0
        self.latent_size = latent_size
        self.max_depth = len(self.blocks)
        self.progression_scale = progression_scale
        self.upsampler = partial(F.interpolate, size=None, scale_factor=progression_scale, mode='linear',
                                 align_corners=True)

    def set_gamma(self, new_gamma):
        for layer in self.self_attention.values():
            layer.gamma = new_gamma

    def do_layer(self, l, h, y=None, last=False):
        if l in self.self_attention:
            h = self.self_attention[l](h)
        h = self.upsampler(h)
        return self.blocks[l](h, y, last)

    def forward(self, z, y=None):
        if self.normalize_latents:
            z = pixel_norm(z)
        h = z.unsqueeze(2)
        h = self.block0(h, y, self.depth == 0)
        if self.depth > 0:
            for i in range(self.depth - 1):
                h = self.do_layer(i, h, y)
            h = self.upsampler(h)
            ult = self.blocks[self.depth - 1](h, y, True)
            if self.alpha < 1.0:
                preult_rgb = self.blocks[self.depth - 2].toRGB(h) if self.depth > 1 else self.block0.toRGB(h)
                return preult_rgb * (1.0 - self.alpha) + ult * self.alpha
            else:
                return ult
        return h


class DBlock(nn.Module):
    def __init__(self, ch_in, ch_out, num_channels, initial_kernel_size=None, is_residual=False,
                 ksize=3, equalized=True, group_size=4, act_alpha=0, spectral=False, sngan_rgb=False, **layer_settings):
        super(DBlock, self).__init__()
        is_last = initial_kernel_size is not None
        self.fromRGB = GeneralConv(num_channels, ch_in, kernel_size=1, equalized=equalized,
                                   act_alpha=-1 if sngan_rgb else act_alpha, spectral=spectral)
        if is_last:
            self.net = [MinibatchStddev(group_size)]
        else:
            self.net = []
        self.net.append(
            GeneralConv(ch_in + (1 if is_last else 0), ch_in, kernel_size=ksize, equalized=equalized,
                        act_alpha=act_alpha, spectral=spectral, **layer_settings))
        self.net.append(
            GeneralConv(ch_in, ch_out, kernel_size=initial_kernel_size if is_last else ksize,
                        pad=0 if is_last else None,
                        equalized=equalized, act_alpha=act_alpha, spectral=spectral, **layer_settings))
        self.net = nn.Sequential(*self.net)
        self.is_last = initial_kernel_size
        if is_residual and not is_last:
            self.residual = nn.Sequential() if ch_in == ch_out else GeneralConv(ch_in, ch_out, kernel_size=1,
                                                                                equalized=equalized, act_alpha=-1,
                                                                                spectral=spectral)
        else:
            self.residual = None

    def forward(self, x, first=False):
        if first:
            x = self.fromRGB(x)
        h = self.net(x)
        if self.residual:
            h = h + self.residual(x)
        return h


class Discriminator(nn.Module):
    def __init__(self, dataset_shape, initial_size, fmap_base, fmap_max, fmap_min, equalized,
                 kernel_size, self_attention_layers, num_classes, sngan_rgb=False, dropout=0.1,
                 do_mode='mul', residual=False, spectral=False, act_norm=None, group_size=4, act_alpha=0):
        super(Discriminator, self).__init__()
        resolution = dataset_shape[-1]
        num_channels = dataset_shape[1]
        progression_scale = 2
        R = int(math.log(resolution, progression_scale))
        assert resolution == progression_scale ** R and resolution >= progression_scale ** initial_size
        self.R = R
        self.progression_scale = progression_scale
        self.initial_depth = initial_size

        def nf(stage):
            return min(max(int(fmap_base / (2.0 ** stage)), fmap_min), fmap_max)

        layer_settings = dict(do=dropout, do_mode=do_mode, act_norm=act_norm)
        initial_kernel_size = progression_scale ** initial_size
        last_block = DBlock(nf(1), nf(0), num_channels, initial_kernel_size=initial_kernel_size, ksize=kernel_size,
                            equalized=equalized, is_residual=residual, group_size=group_size, act_alpha=act_alpha,
                            spectral=spectral, sngan_rgb=sngan_rgb, **layer_settings)
        dummy = []
        self.self_attention = dict()
        for layer in self_attention_layers:
            dummy.append(SelfAttention(nf(layer + 1), spectral))
            self.self_attention[layer] = dummy[-1]
        self.dummy = nn.ModuleList(dummy)
        self.blocks = nn.ModuleList([DBlock(nf(i - initial_size + 2), nf(i - initial_size + 1), num_channels,
                                            ksize=kernel_size, equalized=equalized, initial_kernel_size=None,
                                            is_residual=residual, group_size=group_size, act_alpha=act_alpha,
                                            spectral=spectral, sngan_rgb=sngan_rgb, **layer_settings) for i in
                                     range(R - 1, initial_size - 1, -1)] + [last_block])
        if num_classes != 0:
            self.class_emb = nn.EmbeddingBag(num_classes, nf(initial_size - 2))
            if spectral:
                self.class_emb = spectral_norm(self.class_emb)
        else:
            self.class_emb = None
        self.linear = GeneralConv(nf(0), 1, kernel_size=1, equalized=equalized, pad=None, act_alpha=-1,
                                  spectral=spectral)
        self.depth = 0
        self.alpha = 1.0
        self.max_depth = len(self.blocks) - 1
        self.downsampler = nn.AvgPool1d(kernel_size=progression_scale)

    def set_gamma(self, new_gamma):
        for self_attention_layer in self.self_attention.values():
            self_attention_layer.gamma = new_gamma

    def forward(self, x, y=None):
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
            if (i - 2) in self.self_attention:
                h = self.self_attention[i - 2](h)
        o = self.linear(h).mean(dim=2).squeeze()
        if y:
            o = o + F.sum(self.class_emb(y) * h, axis=1, keepdims=True).mean(dim=2)
        return o, h
