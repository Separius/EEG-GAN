import numpy as np
import math
import torch
from torch import nn
import torch.nn.functional as F
from spectral_norm import spectral_norm as spectral_norm_wrapper
from layers import GeneralConv, Concatenate, ToRGB, SelfAttention, MinibatchStddev, DownSample
from utils import pixel_norm


class GBlock(nn.Module):
    def __init__(self, ch_in, ch_out, num_channels, ksize=3, equalized=True, initial_kernel_size=None, ch_by_ch=False,
                 normalization=None, **layer_settings):
        super(GBlock, self).__init__()
        is_first = initial_kernel_size is not None
        c2 = GeneralConv(ch_out, ch_out, equalized=equalized, ksize=ksize, normalization=normalization,
                         **layer_settings)
        c1 = GeneralConv(ch_in, ch_out, equalized=equalized, ksize=initial_kernel_size if is_first else ksize,
                         pad=initial_kernel_size - 1 if is_first else None, normalization=normalization,
                         **layer_settings)
        self.c1 = c1
        self.c2 = c2
        self.toRGB = ToRGB(ch_out, num_channels, normalization=None if normalization == 'batch_norm' else normalization,
                           ch_by_ch=ch_by_ch, equalized=equalized)

    def forward(self, x, y=None, last=False):
        x = self.c2(self.c1(x, y), y)
        return self.toRGB(x) if last else x


class Generator(nn.Module):
    def __init__(self, progression_scale, dataset_shape, initial_size, fmap_base, fmap_max, fmap_min, kernel_size,
                 equalized, self_attention_layer, num_classes, latent_size=256, upsample='linear',
                 normalize_latents=True, pixelnorm=True, activation='lrelu', dropout=0.1, do_mode='mul',
                 spectral_norm=False, ch_by_ch=False, normalization=None):
        super(Generator, self).__init__()
        resolution = dataset_shape[-1]
        num_channels = dataset_shape[1]
        is_single = not isinstance(progression_scale, (list, tuple))
        if is_single:
            R = int(math.log(resolution, progression_scale))
            assert resolution == progression_scale ** R and resolution >= progression_scale ** initial_size
        else:
            R = len(progression_scale)

        def nf(stage):
            return min(max(int(fmap_base / (2.0 ** stage)), fmap_min), fmap_max)

        if latent_size is None:
            latent_size = nf(0)
        self.normalize_latents = normalize_latents
        layer_settings = dict(pixelnorm=pixelnorm, act=activation, do=dropout, do_mode=do_mode, spectral=spectral_norm,
                              num_classes=num_classes)
        if is_single:
            initial_kernel_size = progression_scale ** initial_size
        else:
            initial_kernel_size = np.prod(progression_scale[:initial_size])
        self.block0 = GBlock(latent_size, nf(1), num_channels, ksize=kernel_size, equalized=equalized,
                             initial_kernel_size=initial_kernel_size, ch_by_ch=ch_by_ch, normalization=normalization,
                             **layer_settings)
        self.self_attention_layer = self_attention_layer
        if self_attention_layer is not None:
            self.self_attention = SelfAttention(nf(self_attention_layer + 1))
        else:
            self.self_attention = None
        self.blocks = nn.ModuleList([GBlock(nf(i - initial_size + 1), nf(i - initial_size + 2), num_channels,
                                            ksize=kernel_size, equalized=equalized, ch_by_ch=ch_by_ch,
                                            normalization=normalization, **layer_settings) for i in
                                     range(initial_size, R)])
        self.depth = 0
        self.alpha = 1.0
        self.latent_size = latent_size
        self.max_depth = len(self.blocks)
        self.progression_scale = progression_scale
        if upsample == 'linear':
            self.upsampler = nn.ModuleList([nn.Upsample(
                scale_factor=progression_scale if is_single else progression_scale[i], mode=upsample,
                align_corners=True) for i in range(initial_size, R)])
        elif upsample == 'nearest':
            self.upsampler = nn.ModuleList(
                [nn.Upsample(scale_factor=progression_scale if is_single else progression_scale[i], mode=upsample) for i
                 in range(initial_size, R)])
        else:
            raise ValueError()

    def set_gamma(self, new_gamma):
        if self.self_attention is not None:
            self.self_attention.gamma = new_gamma

    def consistent_forward(self, z1, z2, stage=2, y=None):
        # NOTE that it assumes that training is finished and alpha=1
        h = torch.cat((z1, z2), dim=0)
        h = h.unsqueeze(2)
        if self.normalize_latents:
            h = pixel_norm(h)
        h = self.block0(h, y, self.depth == 0)
        for i in range(self.depth - 1):
            if i == stage:
                h2 = h[1:, ...]
                h1 = h[0:1, ...].expand_as(h2)
                h = torch.cat((h1, h2), dim=2)
            if i == self.self_attention_layer:
                h = self.self_attention(h)
            h = self.upsampler[i](h)
            h = self.blocks[i](h, y)
        h = self.upsampler[self.depth - 1](h)
        return self.blocks[self.depth - 1](h, y, True)

    def forward(self, x, y=None, intermediate=False):
        h = x.unsqueeze(2)
        if self.normalize_latents:
            h = pixel_norm(h)
        h = self.block0(h, y, self.depth == 0)
        if self.depth > 0 and not intermediate:
            h = self.after_first(h, y)
        return h

    def after_first(self, h, y=None):
        for i in range(self.depth - 1):
            if i == self.self_attention_layer:
                h = self.self_attention(h)
            h = self.upsampler[i](h)
            h = self.blocks[i](h, y)
        h = self.upsampler[self.depth - 1](h)
        ult = self.blocks[self.depth - 1](h, y, True)
        if self.alpha < 1.0:
            if self.depth > 1:
                preult_rgb = self.blocks[self.depth - 2].toRGB(h)
            else:
                preult_rgb = self.block0.toRGB(h)
        else:
            preult_rgb = 0.0
        return preult_rgb * (1.0 - self.alpha) + ult * self.alpha


class DBlock(nn.Module):
    def __init__(self, ch_in, ch_out, num_channels, initial_kernel_size=None, temporal=False, num_stat_channels=1,
                 ksize=3, equalized=True, spectral=False, normalization=None, **layer_settings):
        super(DBlock, self).__init__()
        is_last = initial_kernel_size is not None
        self.fromRGB = GeneralConv(num_channels, ch_in, ksize=1, pixelnorm=False, equalized=equalized,
                                   spectral=spectral,
                                   normalization=None if normalization == 'batch_norm' else normalization)
        if num_stat_channels > 0 and is_last:
            self.net = [MinibatchStddev(temporal, num_stat_channels)]
        else:
            self.net = []
        self.net.append(
            GeneralConv(ch_in + (num_stat_channels if is_last else 0), ch_in, ksize=ksize, equalized=equalized,
                        spectral=spectral, normalization=normalization, **layer_settings))
        self.net.append(
            GeneralConv(ch_in, ch_out, ksize=initial_kernel_size if is_last else ksize, pad=0 if is_last else None,
                        equalized=equalized, spectral=spectral, normalization=normalization, **layer_settings))
        self.net = nn.Sequential(*self.net)

    def forward(self, x, first=False):
        if first:
            x = self.fromRGB(x)
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, progression_scale, dataset_shape, initial_size, fmap_base, fmap_max, fmap_min, equalized,
                 kernel_size, self_attention_layer, num_classes, downsample='average', pixelnorm=False,
                 activation='lrelu', dropout=0.1, do_mode='mul', spectral_norm=False, phase_shuffle=0,
                 temporal_stats=False, num_stat_channels=1, normalization=None):
        super(Discriminator, self).__init__()
        resolution = dataset_shape[-1]
        num_channels = dataset_shape[1]
        is_single = not isinstance(progression_scale, (list, tuple))
        if is_single:
            R = int(math.log(resolution, progression_scale))
            assert resolution == progression_scale ** R and resolution >= progression_scale ** initial_size
            self.R = R
        else:
            self.R = R = len(progression_scale)

        def nf(stage):
            return min(max(int(fmap_base / (2.0 ** stage)), fmap_min), fmap_max)

        layer_settings = dict(pixelnorm=pixelnorm, act=activation, do=dropout, do_mode=do_mode)
        if isinstance(progression_scale, (list, tuple)):
            initial_kernel_size = np.prod(progression_scale[:initial_size])
        else:
            initial_kernel_size = progression_scale ** initial_size
        last_block = DBlock(nf(1), nf(0), num_channels, initial_kernel_size=initial_kernel_size,
                            temporal=temporal_stats, num_stat_channels=num_stat_channels, ksize=kernel_size,
                            equalized=equalized, spectral=spectral_norm, normalization=normalization, **layer_settings)
        layer_settings.update(phase_shuffle=phase_shuffle)
        self.self_attention_layer = self_attention_layer
        if self_attention_layer is not None:
            self.self_attention = SelfAttention(nf(self_attention_layer + 1))
        else:
            self.self_attention = None
        self.blocks = nn.ModuleList([DBlock(nf(i - initial_size + 2), nf(i - initial_size + 1), num_channels,
                                            ksize=kernel_size, equalized=equalized, initial_kernel_size=None,
                                            spectral=spectral_norm, normalization=normalization, **layer_settings) for i
                                     in range(R - 1, initial_size - 1, -1)] + [last_block])
        if num_classes != 0:
            self.class_emb = nn.EmbeddingBag(num_classes, nf(initial_size - 2))
            if spectral_norm:
                self.class_emb = spectral_norm_wrapper(self.class_emb)
        else:
            self.class_emb = None
        self.linear = nn.Linear(nf(0), 1)
        if spectral_norm:
            self.linear = spectral_norm_wrapper(self.linear)
        self.depth = 0
        self.alpha = 1.0
        self.max_depth = len(self.blocks) - 1
        if downsample == 'average':
            self.downsampler = nn.ModuleList(
                [nn.AvgPool1d(kernel_size=progression_scale if is_single else progression_scale[i]) for i in
                 reversed(range(initial_size, R))])
        elif downsample == 'stride':
            self.downsampler = nn.ModuleList(
                [DownSample(scale_factor=progression_scale if is_single else progression_scale[i]) for i in
                 reversed(range(initial_size, R))])

    def set_gamma(self, new_gamma):
        if self.self_attention is not None:
            self.self_attention.gamma = new_gamma

    def consistent_forward(self, x, y=None):
        seq_len = x.size(2)
        step = seq_len // 4
        repeats = seq_len // step - 1
        x = torch.cat([x[..., i * step:i * step + seq_len // 2] for i in range(repeats)], dim=0)
        o, _ = self.forward(x, y)
        return o.view(repeats, -1).mean(dim=0)  # larger value is better

    def forward(self, x, y=None, intermediate=False):
        xhighres = x
        h = self.blocks[-(self.depth + 1)](xhighres, True)
        if self.depth > 0:
            h = self.downsampler[-self.depth](h)
            if self.alpha < 1.0:
                xlowres = self.downsampler[-self.depth](xhighres)
                preult_rgb = self.blocks[-self.depth].fromRGB(xlowres)
                h = h * self.alpha + (1 - self.alpha) * preult_rgb
        for i in range(self.depth, 0, -1):
            h = self.blocks[-i](h)
            if i > 1:
                h = self.downsampler[-i + 1](h)
            if intermediate and i == 2:
                return h
            if (self.self_attention_layer is not None) and i == (self.self_attention_layer + 2):
                h = self.self_attention(h)
        h = h.squeeze(-1)
        o = self.linear(h)
        if y:
            o = o + F.sum(self.class_emb(y) * h, axis=1, keepdims=True)
        return o, h
