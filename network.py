import numpy as np
import math
import torch
from torch import nn
import torch.nn.functional as F
from spectral_norm import spectral_norm as spectral_norm_wrapper
from layers import GeneralConv, SelfAttention, MinibatchStddev, ScaledTanh
from utils import pixel_norm, cudize


class GBlock(nn.Module):
    def __init__(self, ch_in, ch_out, num_channels, ksize=3, equalized=True, initial_kernel_size=None,
                 **layer_settings):
        super(GBlock, self).__init__()
        is_first = initial_kernel_size is not None
        c2 = GeneralConv(ch_out, ch_out, equalized=equalized, ksize=ksize, **layer_settings)
        c1 = GeneralConv(ch_in, ch_out, equalized=equalized, ksize=initial_kernel_size if is_first else ksize,
                         pad=initial_kernel_size - 1 if is_first else None, **layer_settings)
        self.c1 = c1
        self.c2 = c2
        self.toRGB = nn.Sequential(GeneralConv(ch_out, num_channels, ksize=1, act=False, equalized=equalized),
                                   ScaledTanh())

    def forward(self, x, y=None, last=False):
        x = self.c2(self.c1(x, y), y)
        return self.toRGB(x) if last else x


class Generator(nn.Module):
    kernel_middle = None
    kernel_left = None
    kernel_right = None

    def __init__(self, progression_scale, dataset_shape, initial_size, fmap_base, fmap_max, fmap_min, kernel_size,
                 equalized, self_attention_layer, num_classes, depth_offset, latent_size=256, normalize_latents=True,
                 dropout=0.1, do_mode='mul', param_norm=None, act_norm='pixel', is_morph=False):
        super(Generator, self).__init__()
        resolution = dataset_shape[-1]
        num_channels = dataset_shape[1]
        self.depth_offset = depth_offset
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
        layer_settings = dict(do=dropout, do_mode=do_mode, num_classes=num_classes, param_norm=param_norm,
                              act_norm=act_norm)
        if is_single:
            initial_kernel_size = progression_scale ** initial_size
        else:
            initial_kernel_size = np.prod(progression_scale[:initial_size])
        self.block0 = GBlock(latent_size, nf(1), num_channels, ksize=kernel_size, equalized=equalized,
                             initial_kernel_size=initial_kernel_size, **layer_settings)
        self.self_attention_layer = self_attention_layer
        if self_attention_layer is not None:
            self.self_attention = SelfAttention(nf(self_attention_layer + 1))
        else:
            self.self_attention = None
        self.blocks = nn.ModuleList([GBlock(nf(i - initial_size + 1), nf(i - initial_size + 2), num_channels,
                                            ksize=kernel_size, equalized=equalized, **layer_settings)
                                     for i in range(initial_size, R)])
        self.depth = depth_offset
        self.alpha = 1.0
        self.latent_size = latent_size
        self.max_depth = len(self.blocks)
        self.progression_scale = progression_scale
        self.upsampler = nn.ModuleList([nn.Upsample(
            scale_factor=progression_scale if is_single else progression_scale[i], mode='linear', align_corners=True)
            for i in range(initial_size, R)])
        if not is_morph:
            temporal_latent_size = 3 * latent_size // 4
            self.context_maker = nn.Conv1d(temporal_latent_size, temporal_latent_size, kernel_size=3, padding=1)
        else:
            self.context_maker = None

    def set_gamma(self, new_gamma):
        if self.self_attention is not None:
            self.self_attention.gamma = new_gamma

    def do_layer(self, l, h, y=None):
        if l == self.self_attention_layer:
            h = self.self_attention(h)
        h = self.upsampler[l](h)
        return self.blocks[l](h, y)

    @staticmethod
    def generate_kernels(N, C, half):
        tri_up = np.arange(half) / (half - 1)
        tri_down = 1 - tri_up
        Generator.kernel_middle = np.ones((N, C, half * 2), dtype=np.float32)
        Generator.kernel_middle[:, :, :half] = tri_up
        Generator.kernel_middle[:, :, half:] = tri_down
        Generator.kernel_left = np.ones((N, C, half * 2), dtype=np.float32)
        Generator.kernel_left[:, :, half:] = tri_down
        Generator.kernel_right = np.ones((N, C, half * 2), dtype=np.float32)
        Generator.kernel_right[:, :, :half] = tri_up
        Generator.kernel_middle = cudize(torch.from_numpy(Generator.kernel_middle))
        Generator.kernel_right = cudize(torch.from_numpy(Generator.kernel_right))
        Generator.kernel_left = cudize(torch.from_numpy(Generator.kernel_left))

    @staticmethod
    def morph(h, seq_len, num_z):
        N, C, _ = h.size()
        half = seq_len // 2
        res = torch.autograd.Variable(cudize(torch.zeros(N, C, (num_z + 1) * half)))
        if Generator.kernel_middle is None or Generator.kernel_middle.size(0) != N:
            Generator.generate_kernels(N, C, half)
        for i in range(num_z):
            x = h[:, :, i * seq_len:(i + 1) * seq_len]
            if i == 0 and num_z > 1:
                k = Generator.kernel_left
            elif i == (num_z - 1) and num_z > 1:
                k = Generator.kernel_right
            else:
                k = Generator.kernel_middle
            res[:, :, half * i:half * (i + 2)] = k * x
        return res

    def forward(self, z_global, z_temporal=None, y=None):  # glob=(N,z_dim) or glob=(N,z/4), temporal=(N,3z/4,T)
        if self.depth < self.depth_offset:
            raise ValueError()
        if self.normalize_latents:
            z_global = pixel_norm(z_global)
        if z_global.dim() == 2:
            z_global = z_global.unsqueeze(2)
        if z_temporal is None:
            h = z_global
        else:
            if self.normalize_latents:
                z_temporal = pixel_norm(z_temporal)
            if self.context_maker is not None:
                z_temporal = self.context_maker(z_temporal)
            if z_global.size(2) != z_temporal.size(2):
                z_global = z_global.repeat(1, 1, z_temporal.size(2))
            z = torch.cat((z_global, z_temporal), dim=1)
            h = z.permute(0, 2, 1).contiguous().view(-1, z.size(1), 1)
        h = self.block0(h, y, self.depth == 0)
        for i in range(self.depth_offset):
            h = self.do_layer(i, h, y)
        if z_temporal is not None:
            h = h.permute(0, 2, 1).contiguous().view(z.size(0), -1, h.size(1)).permute(0, 2, 1)
            if self.context_maker is None:
                num_z = z_temporal.size(2)
                h = self.morph(h, h.size(2) // num_z, num_z)
        if self.depth == self.depth_offset:
            return h
        for i in range(self.depth_offset, self.depth - 1):
            h = self.do_layer(i, h, y)
        h = self.upsampler[self.depth - 1](h)
        ult = self.blocks[self.depth - 1](h, y, True)
        if self.alpha < 1.0:
            preult_rgb = self.blocks[self.depth - 2].toRGB(h) if self.depth > 1 else self.block0.toRGB(h)
            return preult_rgb * (1.0 - self.alpha) + ult * self.alpha
        else:
            return ult


class DBlock(nn.Module):
    def __init__(self, ch_in, ch_out, num_channels, initial_kernel_size=None, temporal=False, num_stat_channels=1,
                 ksize=3, equalized=True, param_norm=None, **layer_settings):
        super(DBlock, self).__init__()
        is_last = initial_kernel_size is not None
        self.fromRGB = GeneralConv(num_channels, ch_in, ksize=1, equalized=equalized, param_norm=param_norm)
        if num_stat_channels > 0 and is_last:
            self.net = [MinibatchStddev(temporal, num_stat_channels)]
        else:
            self.net = []
        self.net.append(
            GeneralConv(ch_in + (num_stat_channels if is_last else 0), ch_in, ksize=ksize, equalized=equalized,
                        param_norm=param_norm, **layer_settings))
        self.net.append(
            GeneralConv(ch_in, ch_out, ksize=initial_kernel_size if is_last else ksize, pad=0 if is_last else None,
                        equalized=equalized, param_norm=param_norm, **layer_settings))
        self.net = nn.Sequential(*self.net)
        self.is_last = initial_kernel_size

    def forward(self, x, first=False):
        if first:
            x = self.fromRGB(x)
        return self.net(x)


# TODO am i running the self attention in the right place (in respect to sa.start, sa.layer)
class Discriminator(nn.Module):
    def __init__(self, progression_scale, dataset_shape, initial_size, fmap_base, fmap_max, fmap_min, equalized,
                 kernel_size, self_attention_layer, num_classes, depth_offset, dropout=0.1, do_mode='mul',
                 param_norm=None, temporal_stats=False, num_stat_channels=1, act_norm=None, calc_std=False):
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

        layer_settings = dict(do=dropout, do_mode=do_mode, act_norm=act_norm)
        if not is_single:
            initial_kernel_size = np.prod(progression_scale[:initial_size])
            self.final_receptive_field = np.prod(progression_scale[:initial_size + depth_offset])
        else:
            initial_kernel_size = progression_scale ** initial_size
            self.final_receptive_field = progression_scale ** (initial_size + depth_offset)
        last_block = DBlock(nf(1), nf(0), num_channels, initial_kernel_size=initial_kernel_size,
                            temporal=temporal_stats, num_stat_channels=num_stat_channels, ksize=kernel_size,
                            equalized=equalized, param_norm=param_norm, **layer_settings)
        self.self_attention_layer = self_attention_layer
        if self_attention_layer is not None:
            self.self_attention = SelfAttention(nf(self_attention_layer + 1))
        else:
            self.self_attention = None
        self.blocks = nn.ModuleList([DBlock(nf(i - initial_size + 2), nf(i - initial_size + 1), num_channels,
                                            ksize=kernel_size, equalized=equalized, initial_kernel_size=None,
                                            param_norm=param_norm, **layer_settings) for i in
                                     range(R - 1, initial_size - 1, -1)] + [last_block])
        if num_classes != 0:
            self.class_emb = nn.EmbeddingBag(num_classes, nf(initial_size - 2))
            if param_norm == 'spectral':
                self.class_emb = spectral_norm_wrapper(self.class_emb)
        else:
            self.class_emb = None
        self.linear = nn.Conv1d(nf(0), 1, kernel_size=1)
        if param_norm == 'spectral':
            self.linear = spectral_norm_wrapper(self.linear)
        self.depth = depth_offset
        self.alpha = 1.0
        self.max_depth = len(self.blocks) - 1
        self.downsampler = nn.ModuleList(
            [nn.AvgPool1d(kernel_size=progression_scale if is_single else progression_scale[i]) for i in
             reversed(range(initial_size, R))])
        self.depth_offset = depth_offset
        if calc_std:
            self.calc_std = nn.Conv1d(nf(0), nf(0) // 16, kernel_size=1)
        else:
            self.calc_std = None

    def set_gamma(self, new_gamma):
        if self.self_attention is not None:
            self.self_attention.gamma = new_gamma

    def forward(self, x, y=None):
        if self.depth < self.depth_offset:
            raise ValueError()
        xhighres = x
        if self.depth == self.depth_offset:
            step = self.final_receptive_field // 4
            t = int(math.floor((xhighres.size(2) - self.final_receptive_field) / step + 1))
            i = self.depth + 1
            hs = [self.blocks[-i](xhighres[:, :, j * step:j * step + self.final_receptive_field], True) for j in
                  range(t)]
            is_list = True
        else:
            hs = (self.blocks[-(self.depth + 1)](xhighres, True),)
            is_list = False
        res = []
        for h in hs:
            if self.depth > 0:
                h = self.downsampler[-self.depth](h)
                if self.alpha < 1.0:
                    xlowres = self.downsampler[-self.depth](xhighres)
                    preult_rgb = self.blocks[-self.depth].fromRGB(xlowres)
                    h = h * self.alpha + (1 - self.alpha) * preult_rgb
            res.append(h)
        for i in range(self.depth, self.depth_offset + 1, -1):
            h = self.blocks[-i](h)
            if i > 1:
                h = self.downsampler[-i + 1](h)
            if (self.self_attention_layer is not None) and i == (self.self_attention_layer + 2):
                h = self.self_attention(h)
        i = self.depth_offset + 1
        if is_list:
            ans = res
        else:
            step = self.final_receptive_field // 4
            t = int(math.floor((h.size(2) - self.final_receptive_field) / step + 1))
            ans = [self.blocks[-i](h[:, :, j * step:j * step + self.final_receptive_field]) for j in range(t)]
        res = []
        for h in ans:
            if self.depth_offset > 1 and self.depth_offset != self.depth:
                h = self.downsampler[-self.depth_offset](h)
            if (self.self_attention_layer is not None) and self.depth_offset == (self.self_attention_layer + 1):
                h = self.self_attention(h)
            for i in range(self.depth_offset, 0, -1):
                h = self.blocks[-i](h)
                if i > 1:
                    h = self.downsampler[-i + 1](h)
                if (self.self_attention_layer is not None) and i == (self.self_attention_layer + 2):
                    h = self.self_attention(h)
            res.append(h)
        assert res[0].shape[2] == 1
        h = torch.cat(res, dim=2)
        o = self.linear(h).mean(dim=2).squeeze()
        if self.calc_std is not None:
            o = o + self.calc_std(h).std(dim=2).mean(dim=1)
        if y:
            o = o + F.sum(self.class_emb(y) * h, axis=1, keepdims=True).mean(dim=2)
        return o, h
