import math
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from utils import pixel_norm, cudize
from layers import GeneralConv, SelfAttention, MinibatchStddev, ScaledTanh, spectral_norm


class GBlock(nn.Module):
    def __init__(self, ch_in, ch_out, num_channels, ksize=3, equalized=True, initial_kernel_size=None,
                 is_residual=False, **layer_settings):
        super(GBlock, self).__init__()
        is_first = initial_kernel_size is not None
        self.c1 = GeneralConv(ch_in, ch_out, equalized=equalized, ksize=initial_kernel_size if is_first else ksize,
                              pad=initial_kernel_size - 1 if is_first else None, **layer_settings)
        self.c2 = GeneralConv(ch_out, ch_out, equalized=equalized, ksize=ksize, **layer_settings)
        self.toRGB = nn.Sequential(GeneralConv(ch_out, num_channels, ksize=1, act=False, equalized=equalized),
                                   ScaledTanh())
        if not is_first and is_residual:
            self.residual = nn.Sequential() if ch_in == ch_out else nn.Conv1d(ch_in, ch_out, kernel_size=1)
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
    kernel_middle = None
    kernel_left = None
    kernel_right = None

    def __init__(self, progression_scale, dataset_shape, initial_size, fmap_base, fmap_max, fmap_min, kernel_size,
                 equalized, self_attention_layers, num_classes, depth_offset, is_extended, latent_size=256,
                 residual=False, normalize_latents=True, dropout=0.1, do_mode='mul', param_norm=None, act_norm='pixel',
                 is_morph=False, extend_context_size=3, use_extend_padding=False):
        super(Generator, self).__init__()
        resolution = dataset_shape[-1]
        num_channels = dataset_shape[1]
        self.depth_offset = depth_offset
        self.extend_context_size = extend_context_size
        # TODO this is not properly handled(in prepare_g_data)
        self.use_extend_padding = use_extend_padding
        R = int(math.log(resolution, progression_scale))
        assert resolution == progression_scale ** R and resolution >= progression_scale ** initial_size

        def nf(stage):
            return min(max(int(fmap_base / (2.0 ** stage)), fmap_min), fmap_max)

        if latent_size is None:
            latent_size = nf(0)
        self.normalize_latents = normalize_latents
        layer_settings = dict(do=dropout, do_mode=do_mode, num_classes=num_classes, param_norm=param_norm,
                              act_norm=act_norm)
        initial_kernel_size = progression_scale ** initial_size
        self.block0 = GBlock(latent_size, nf(1), num_channels, ksize=kernel_size, equalized=equalized,
                             initial_kernel_size=initial_kernel_size, is_residual=residual, **layer_settings)
        self.self_attention = dict()
        for layer in self_attention_layers:
            self.self_attention[layer] = SelfAttention(nf(layer + 1))
        self.blocks = nn.ModuleList([GBlock(nf(i - initial_size + 1), nf(i - initial_size + 2), num_channels,
                                            ksize=kernel_size, equalized=equalized, is_residual=residual,
                                            **layer_settings) for i in range(initial_size, R)])
        self.depth = depth_offset
        self.alpha = 1.0
        self.latent_size = latent_size
        self.max_depth = len(self.blocks)
        self.progression_scale = progression_scale
        self.upsampler = nn.Upsample(scale_factor=progression_scale, mode='linear', align_corners=True)
        self.is_morph = is_morph
        if extend_context_size != 0 and self.is_extended:
            temporal_latent_size = 3 * latent_size // 4
            self.context_maker = nn.Conv1d(latent_size, temporal_latent_size, kernel_size=extend_context_size,
                                           padding=extend_context_size // 2 if use_extend_padding else 0)
        else:
            self.context_maker = None
        self.is_extended = is_extended

    def set_gamma(self, new_gamma):
        for layer in self.self_attention.values():
            layer.gamma = new_gamma

    def do_layer(self, l, h, y=None):
        if l in self.self_attention_layer:
            h = self.self_attention[l](h)
        h = self.upsampler(h)
        return self.blocks[l](h, y)

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
            h = self.do_layer(i, h, y)
        h = self.upsampler[self.depth - 1](h)
        return self.blocks[self.depth - 1](h, y, True)

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
        half = max(seq_len // 2, 1)
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

    def static_extend(self, z, y=None, concatenation_layer=0, morph=False):
        N, z_dim, T = z.size()
        if self.normalize_latents:
            z = pixel_norm(z)
        h = z.permute(0, 2, 1).contiguous().view(-1, z.size(1), 1)
        h = self.block0(h, y, self.depth == 0)
        for i in range(concatenation_layer):
            h = self.do_layer(i, h, y)
        h = h.permute(0, 2, 1).contiguous().view(z.size(0), -1, h.size(1)).permute(0, 2, 1)
        if morph:
            h = self.morph(h, h.size(2) // T, T)
        if self.depth == concatenation_layer:
            return h
        for i in range(concatenation_layer, self.depth - 1):
            h = self.do_layer(i, h, y)
        h = self.upsampler(h)
        ult = self.blocks[self.depth - 1](h, y, True)
        if self.alpha < 1.0:
            preult_rgb = self.blocks[self.depth - 2].toRGB(h) if self.depth > 1 else self.block0.toRGB(h)
            return preult_rgb * (1.0 - self.alpha) + ult * self.alpha
        else:
            return ult

    def forward(self, z_global, z_temporal=None, y=None):  # glob=(N,z_dim) or glob=(N,z/4,[T]), temporal=(N,3z/4,T)
        if self.is_extended and (self.depth < self.depth_offset):
            raise ValueError()
        if self.normalize_latents:
            z_global = pixel_norm(z_global)
        if z_global.dim() == 2:
            z_global = z_global.unsqueeze(2)
        if z_temporal is None:
            h = z_global
            if self.is_extended:
                raise ValueError()
        else:
            if not self.is_extended:
                raise ValueError()
            if self.normalize_latents:
                z_temporal = pixel_norm(z_temporal)
            if z_global.size(2) != z_temporal.size(2):
                z_global = z_global.repeat(1, 1, z_temporal.size(2))
            if self.context_maker is not None:
                z = torch.cat((z_global, z_temporal), dim=1)
                z_temporal = self.context_maker(z)
            z = torch.cat((z_global, z_temporal), dim=1)
            h = z.permute(0, 2, 1).contiguous().view(-1, z.size(1), 1)
        h = self.block0(h, y, self.depth == 0)
        for i in range(self.depth_offset):
            h = self.do_layer(i, h, y)
        if z_temporal is not None:
            h = h.permute(0, 2, 1).contiguous().view(z.size(0), -1, h.size(1)).permute(0, 2, 1)
            if self.is_morph:
                num_z = z_temporal.size(2)
                if not self.use_extend_padding:
                    num_z -= self.extend_context_size - 1
                h = self.morph(h, h.size(2) // num_z, num_z)
        if self.depth == self.depth_offset:
            return h
        for i in range(self.depth_offset, self.depth - 1):
            h = self.do_layer(i, h, y)
        h = self.upsampler(h)
        ult = self.blocks[self.depth - 1](h, y, True)
        if self.alpha < 1.0:
            preult_rgb = self.blocks[self.depth - 2].toRGB(h) if self.depth > 1 else self.block0.toRGB(h)
            return preult_rgb * (1.0 - self.alpha) + ult * self.alpha
        else:
            return ult


class DBlock(nn.Module):
    def __init__(self, ch_in, ch_out, num_channels, initial_kernel_size=None, temporal=False, is_residual=False,
                 num_stat_channels=1, ksize=3, equalized=True, param_norm=None, **layer_settings):
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
        if is_residual and not is_last:
            self.residual = nn.Sequential() if ch_in == ch_out else nn.Conv1d(ch_in, ch_out, kernel_size=1)
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
    def __init__(self, progression_scale, dataset_shape, initial_size, fmap_base, fmap_max, fmap_min, equalized,
                 kernel_size, self_attention_layers, num_classes, depth_offset, dropout=0.1, do_mode='mul',
                 residual=False, param_norm=None, temporal_stats=False, num_stat_channels=1, act_norm=None,
                 calc_std=False, window_from_first=False):
        super(Discriminator, self).__init__()
        resolution = dataset_shape[-1]
        num_channels = dataset_shape[1]
        R = int(math.log(resolution, progression_scale))
        assert resolution == progression_scale ** R and resolution >= progression_scale ** initial_size
        self.R = R
        self.progression_scale = progression_scale
        self.initial_depth = initial_size
        self.window_from_first = window_from_first
        depth_offset = None if window_from_first else depth_offset

        def nf(stage):
            return min(max(int(fmap_base / (2.0 ** stage)), fmap_min), fmap_max)

        layer_settings = dict(do=dropout, do_mode=do_mode, act_norm=act_norm)
        initial_kernel_size = progression_scale ** initial_size
        if depth_offset is not None:
            self.final_receptive_field = progression_scale ** (initial_size + depth_offset)
        else:
            self.final_receptive_field = None
        last_block = DBlock(nf(1), nf(0), num_channels, initial_kernel_size=initial_kernel_size,
                            temporal=temporal_stats, num_stat_channels=num_stat_channels, ksize=kernel_size,
                            equalized=equalized, param_norm=param_norm, is_residual=residual, **layer_settings)
        self.self_attention = {self_attention_layer: SelfAttention(nf(self_attention_layer + 1)) for
                               self_attention_layer in self_attention_layers}
        self.blocks = nn.ModuleList([DBlock(nf(i - initial_size + 2), nf(i - initial_size + 1), num_channels,
                                            ksize=kernel_size, equalized=equalized, initial_kernel_size=None,
                                            param_norm=param_norm, is_residual=residual, **layer_settings) for i in
                                     range(R - 1, initial_size - 1, -1)] + [last_block])
        if num_classes != 0:
            self.class_emb = nn.EmbeddingBag(num_classes, nf(initial_size - 2))
            if param_norm == 'spectral':
                self.class_emb = spectral_norm(self.class_emb)
        else:
            self.class_emb = None
        self.linear = nn.Conv1d(nf(0), 1, kernel_size=1)
        if param_norm == 'spectral':
            self.linear = spectral_norm(self.linear)
        self.depth = depth_offset if depth_offset is not None else 0
        self.alpha = 1.0
        self.max_depth = len(self.blocks) - 1
        self.downsampler = nn.AvgPool1d(kernel_size=progression_scale)
        self.depth_offset = depth_offset
        if calc_std:
            self.calc_std = nn.Conv1d(nf(0), nf(0) // 4, kernel_size=1)
        else:
            self.calc_std = None

    def consistent_forward(self, x, y=None):
        seq_len = x.size(2)
        step = seq_len // 4
        repeats = seq_len // step - 1
        x = torch.cat([x[..., i * step:i * step + seq_len // 2] for i in range(repeats)], dim=0)
        o, _ = self.forward(x, y)
        return o.view(repeats, -1).mean(dim=0)

    def set_gamma(self, new_gamma):
        for self_attention_layer in self.self_attention.values():
            self_attention_layer.gamma = new_gamma

    def forward(self, x, y=None):
        if self.depth_offset is not None and (self.depth < self.depth_offset):
            raise ValueError()
        xhighres = x
        if (self.depth_offset is None) or (self.depth == self.depth_offset):
            if self.final_receptive_field is not None:
                step = self.final_receptive_field // 4
            else:
                step = self.progression_scale ** (self.depth + self.initial_depth) // 4
            t = int(math.floor((xhighres.size(2) - step * 4) / step + 1))
            i = self.depth + 1
            hs = [self.blocks[-i](xhighres[:, :, j * step:j * step + step * 4], True) for j in range(t)]
            is_list = True
        else:
            hs = (self.blocks[-(self.depth + 1)](xhighres, True),)
            is_list = False
        res = []
        for h in hs:
            if self.depth > 0:
                h = self.downsampler(h)
                if self.alpha < 1.0:
                    xlowres = self.downsampler(xhighres)
                    preult_rgb = self.blocks[-self.depth].fromRGB(xlowres)
                    h = h * self.alpha + (1 - self.alpha) * preult_rgb
            res.append(h)
        depth_offset = self.depth if self.depth_offset is None else self.depth_offset
        for i in range(self.depth, depth_offset + 1, -1):
            h = self.blocks[-i](h)
            if i > 1:
                h = self.downsampler(h)
            if (i - 2) in self.self_attention:
                h = self.self_attention[i - 2](h)
        i = depth_offset + 1
        if is_list:
            ans = res
        else:
            step = max(self.final_receptive_field // 4, 1)
            t = int(math.floor((h.size(2) - self.final_receptive_field) / step + 1))
            ans = [self.blocks[-i](h[:, :, j * step:j * step + self.final_receptive_field]) for j in range(t)]
        res = []
        for h in ans:
            if depth_offset > 1 and depth_offset != self.depth:
                h = self.downsampler(h)
            if (depth_offset - 1) in self.self_attention:
                h = self.self_attention[depth_offset - 1](h)
            for i in range(depth_offset, 0, -1):
                h = self.blocks[-i](h)
                if i > 1:
                    h = self.downsampler(h)
                if (i - 2) in self.self_attention:
                    h = self.self_attention[i - 2](h)
            res.append(h)
        assert res[0].shape[2] == 1
        h = torch.cat(res, dim=2)
        o = self.linear(h).mean(dim=2).squeeze()
        if self.calc_std is not None:
            o = o - self.calc_std(h).std(dim=2).mean(dim=1)
        if y:
            o = o + F.sum(self.class_emb(y) * h, axis=1, keepdims=True).mean(dim=2)
        return o, h
