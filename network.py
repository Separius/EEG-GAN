from typing import Optional

import torch
from torch import nn
from torch.nn.utils import spectral_norm

from utils import pixel_norm, resample_signal
from layers import GeneralConv, SelfAttention, MinibatchStddev, ScaledTanh, PassChannelResidual, ConcatResidual


class GBlock(nn.Module):
    # layer_settings = dict(z_to_bn_size=0, equalized=True, spectral=False, init='kaiming_normal',
    #           act_alpha=0.2, do=0, num_classes=0, act_norm=None, bias=True, separable=False)
    def __init__(self, ch_in, ch_out, ch_rgb, k_size=3, initial_kernel_size=None, is_residual=False, no_tanh=False,
                 deep=False, per_channel_noise=False, to_rgb_mode='pggan', **layer_settings):
        super().__init__()
        is_first = initial_kernel_size is not None
        first_k_size = initial_kernel_size if is_first else k_size
        hidden_size = (ch_in // 4) if deep else ch_out
        self.c1 = GeneralConv(ch_in, hidden_size, kernel_size=first_k_size,
                              pad=initial_kernel_size - 1 if is_first else None, **layer_settings)
        self.c2 = GeneralConv(hidden_size, hidden_size, kernel_size=k_size, **layer_settings)
        if per_channel_noise:
            self.c1_noise_weight = nn.Parameter(torch.zeros(1, hidden_size, 1))
            self.c2_noise_weight = nn.Parameter(torch.zeros(1, hidden_size, 1))
        else:
            self.c1_noise_weight, self.c2_noise_weight = None, None
        if deep:
            self.c3 = GeneralConv(hidden_size, hidden_size, kernel_size=k_size, **layer_settings)
            self.c4 = GeneralConv(hidden_size, ch_out, kernel_size=k_size, **layer_settings)
            if per_channel_noise:
                self.c3_noise_weight = nn.Parameter(torch.zeros(1, hidden_size, 1))
                self.c4_noise_weight = nn.Parameter(torch.zeros(1, ch_out, 1))
            else:
                self.c3_noise_weight, self.c4_noise_weight = None, None
        reduced_layer_settings = dict(equalized=layer_settings['equalized'], spectral=layer_settings['equalized'],
                                      init=layer_settings['equalized'])
        if to_rgb_mode == 'pggan':
            to_rgb = GeneralConv(ch_out, ch_rgb, kernel_size=1, act_alpha=-1, **reduced_layer_settings)
        elif to_rgb_mode in {'sngan', 'sagan'}:
            to_rgb = GeneralConv(ch_out, ch_rgb if to_rgb_mode == 'sngan' else ch_out, kernel_size=3, act_alpha=0.2,
                                 **reduced_layer_settings)
            if to_rgb_mode == 'sagan':
                to_rgb = nn.Sequential(
                    GeneralConv(ch_out, ch_rgb, kernel_size=1, act_alpha=-1, **reduced_layer_settings), to_rgb)
        elif to_rgb_mode == 'biggan':
            to_rgb = nn.Sequential(nn.BatchNorm1d(ch_out), nn.ReLU(),
                                   GeneralConv(ch_out, ch_rgb, kernel_size=3, act_alpha=-1, **reduced_layer_settings))
        else:
            raise ValueError()
        if no_tanh:
            self.toRGB = to_rgb
        else:
            self.toRGB = nn.Sequential(to_rgb, ScaledTanh())
        if deep:
            self.residual = PassChannelResidual()
        else:
            if not is_first and is_residual:
                self.residual = nn.Sequential() if ch_in == ch_out else \
                    GeneralConv(ch_in, ch_out, 1, act_alpha=-1, **reduced_layer_settings)
            else:
                self.residual = None
        self.deep = deep

    @staticmethod
    def get_per_channel_noise(noise_weight):
        return None if noise_weight is None else torch.randn(*noise_weight.size()) * noise_weight

    def forward(self, x, y=None, z=None, last=False):
        h = self.c1(x, y=y, z=z, conv_noise=self.get_per_channel_noise(self.c1_noise_weight))
        h = self.c2(h, y=y, z=z, conv_noise=self.get_per_channel_noise(self.c2_noise_weight))
        if self.deep:
            h = self.c3(h, y=y, z=z, conv_noise=self.get_per_channel_noise(self.c3_noise_weight))
            h = self.c4(h, y=y, z=z, conv_noise=self.get_per_channel_noise(self.c4_noise_weight))
            h = self.residual(h, x)
        elif self.residual is not None:
            h = h + self.residual(x)
        if last:
            return self.toRGB(h)
        return h


class NeoGenerator(nn.Module):
    def __init__(self, initial_kernel_size, num_rgb_channels, fmap_base, fmap_max, fmap_min, kernel_size,
                 self_attention_layers, progression_scale_up, progression_scale_down, residual, separable, equalized,
                 spectral, init, act_alpha, z_distribution, latent_size=256, no_tanh=False, deep=False,
                 per_channel_noise=False, to_rgb_mode='pggan', z_to_bn=False, split_z=False, dropout=0.2, num_classes=0,
                 act_norm='pixel', conv_only=False, shared_embedding_size=32, concat_conditioning=False,
                 normalize_latents=True):
        super().__init__()
        R = len(progression_scale_up)
        self.progression_scale_up = progression_scale_up
        self.progression_scale_down = progression_scale_down
        self.depth = 0
        self.alpha = 1.0
        self.split_z = split_z
        self.z_distribution = z_distribution
        self.concat_conditioning = concat_conditioning
        self.conv_only = conv_only
        self.initial_kernel_size = initial_kernel_size
        self.normalize_latents = normalize_latents

        def nf(stage):
            return min(max(int(fmap_base / (2.0 ** stage)), fmap_min), fmap_max)

        if latent_size is None:
            latent_size = nf(0)
        self.input_latent_size = latent_size
        if num_classes != 0:
            if shared_embedding_size > 0:
                self.y_encoder = GeneralConv(num_classes, shared_embedding_size, kernel_size=1,
                                             equalized=False, act_alpha=act_alpha, spectral=False, bias=False)
                num_classes = shared_embedding_size
            else:
                self.y_encoder = nn.Sequential()
        else:
            self.y_encoder = None
        if split_z:
            latent_size //= R + 2  # we also give part of the z to the first layer
        self.latent_size = latent_size
        if concat_conditioning:
            extra_input = latent_size + num_classes
            if act_norm == 'batch':  # we can only have vanilla BN now
                z_to_bn = False
                num_classes = 0
        else:
            extra_input = 0
        block_settings = dict(ch_rgb=num_rgb_channels, k_size=kernel_size, is_residual=residual,
                              no_tanh=no_tanh, deep=deep, per_channel_noise=per_channel_noise, to_rgb_mode=to_rgb_mode)
        layer_settings = dict(z_to_bn_size=latent_size if z_to_bn else 0, equalized=equalized, spectral=spectral,
                              init=init, act_alpha=act_alpha, do=dropout, num_classes=num_classes, act_norm=act_norm,
                              bias=True, separable=separable)
        self.block0 = GBlock(latent_size + extra_input, nf(1),
                             initial_kernel_size=kernel_size if conv_only else initial_kernel_size, **block_settings,
                             **layer_settings)
        dummy = []  # to make SA layers registered
        self.self_attention = dict()
        for layer in self_attention_layers:
            dummy.append(SelfAttention(nf(layer + 1), spectral, init))
            self.self_attention[layer] = dummy[-1]
        self.dummy = nn.ModuleList(dummy)
        self.blocks = nn.ModuleList(
            [GBlock(nf(i + 1) + extra_input, nf(i + 2), **block_settings, **layer_settings) for i in range(R)])
        self.max_depth = len(self.blocks)

    def _cat_z(self, l, y, z):
        if not self.split_z:
            return y
        z_slice = z[:, (2 + l) * self.latent_size:(3 + l) * self.latent_size, None]
        return z_slice if y is None else torch.cat([y, z_slice.repeat(1, 1, y.shape[2])], dim=1)

    def do_layer(self, l, h, y, z):
        if l in self.self_attention:
            h, attention_map = self.self_attention[l](h)
        else:
            attention_map = None
        h = resample_signal(h, self.progression_scale_down[l], self.progression_scale_up[l], True)
        return self.blocks[l](h, self._cat_z(l, y, z), False), attention_map

    def forward(self, z, y=None):
        if y is not None:
            if y.ndimension() == 2:
                y = y.unsqueeze(2)
            if self.y_encoder is not None:
                y = self.y_encoder(y)
            else:
                y = None
        # WIP
        if self.normalize_latents:
            z = pixel_norm(z)
        h = z.unsqueeze(2)
        if self.split_z:
            h = h[:, :self.latent_size, :]
        h = self.block0(h, self._cat_z(-1, y, z), self.depth == 0)
        if self.depth == 0:
            return h, {}
        all_attention_maps = {}
        for i in range(self.depth - 1):
            h, attention_map = self.do_layer(i, h, y, z)
            if attention_map is not None:
                all_attention_maps[i] = attention_map
        h = resample_signal(h, self.progression_scale_down[self.depth - 1], self.progression_scale_up[self.depth - 1],
                            True)
        ult = self.blocks[self.depth - 1](h, self._cat_z(self.depth - 1, y, z), True)
        if self.alpha == 1.0:
            return ult, all_attention_maps
        preult_rgb = self.blocks[self.depth - 2].toRGB(h) if self.depth > 1 else self.block0.toRGB(h)
        return preult_rgb * (1.0 - self.alpha) + ult * self.alpha, all_attention_maps


class DBlock(nn.Module):
    # layer_settings = dict(equalized=True, spectral=False, init='kaiming_normal',
    #           act_alpha=0.2, do=0, num_classes=0, act_norm=None, bias=True, separable=False)
    def __init__(self, ch_in, ch_out, ch_rgb, k_size=3, initial_kernel_size=None, is_residual=False,
                 deep=False, group_size=4, temporal_groups_per_window=1, conv_disc=False, **layer_settings):
        super().__init__()
        is_last = initial_kernel_size is not None
        self.net = []
        if is_last:
            self.net.append(MinibatchStddev(group_size, temporal_groups_per_window, initial_kernel_size))
        hidden_size = (ch_out // 4) if deep else ch_in
        self.net.append(
            GeneralConv(ch_in + (1 if is_last else 0), hidden_size, kernel_size=k_size, **layer_settings))
        is_linear_last = is_last and not conv_disc
        self.net.append(GeneralConv(hidden_size, hidden_size if deep else ch_out,
                                    kernel_size=initial_kernel_size if is_linear_last else k_size,
                                    pad=0 if is_linear_last else None, **layer_settings))
        if deep:
            self.net.append(
                GeneralConv(hidden_size, hidden_size, kernel_size=k_size, **layer_settings))
            self.net.append(
                GeneralConv(hidden_size, ch_out, kernel_size=k_size, **layer_settings))
        self.net = nn.Sequential(*self.net)
        self.is_last = initial_kernel_size
        reduced_layer_settings = dict(equalized=layer_settings['equalized'], spectral=layer_settings['equalized'],
                                      init=layer_settings['equalized'])
        self.fromRGB = GeneralConv(ch_rgb, ch_in, kernel_size=1, act_alpha=layer_settings['act_alpha'],
                                   **reduced_layer_settings)
        if deep:
            self.residual = ConcatResidual(ch_in, ch_out, **reduced_layer_settings)
        else:
            if is_residual and not is_last:
                self.residual = nn.Sequential() if ch_in == ch_out else GeneralConv(ch_in, ch_out, kernel_size=1,
                                                                                    act_alpha=-1,
                                                                                    **reduced_layer_settings)
            else:
                self.residual = None
        self.deep = deep

    def forward(self, x, first=False):
        if first:
            x = self.fromRGB(x)
        h = self.net(x)
        if self.deep:
            return self.residual(x, h)
        if self.residual:
            h = h + self.residual(x)
        return h


class Discriminator(nn.Module):
    def __init__(self, initial_kernel_size, num_channels, fmap_base, fmap_max, fmap_min, kernel_size, equalized,
                 self_attention_layers, progression_scale_up, progression_scale_down, init, act_alpha,
                 residual, sagan_non_local, factorized_attention, global_conds, temporal_conds, sngan_rgb: bool = False,
                 dropout: float = 0.2, spectral: bool = False, act_norm: Optional[str] = None, group_size: int = 4):
        # temporal_conds = {'temporal_1': 3, 'temporal_2': 1, ...}
        super().__init__()
        R = len(progression_scale_up)
        self.progression_scale_up = progression_scale_up
        self.progression_scale_down = progression_scale_down

        def nf(stage):
            return min(max(int(fmap_base / (2.0 ** stage)), fmap_min), fmap_max)

        layer_settings = dict(do=dropout, act_norm=act_norm)
        last_block = DBlock(nf(1), nf(0), num_channels, initial_kernel_size=initial_kernel_size, ksize=kernel_size,
                            equalized=equalized, is_residual=residual, group_size=group_size, act_alpha=act_alpha,
                            spectral=spectral, sngan_rgb=sngan_rgb, init=init, **layer_settings)
        dummy = []  # to make SA layers registered
        self.self_attention = dict()
        for layer in self_attention_layers:
            dummy.append(SelfAttention(nf(layer + 1), sagan_non_local, spectral, factorized_attention, init))
            self.self_attention[layer] = dummy[-1]
        self.dummy = nn.ModuleList(dummy)
        self.blocks = nn.ModuleList([DBlock(nf(i + 2), nf(i + 1), num_channels, ksize=kernel_size, equalized=equalized,
                                            initial_kernel_size=None, is_residual=residual, group_size=group_size,
                                            act_alpha=act_alpha, spectral=spectral, init=init, sngan_rgb=sngan_rgb,
                                            **layer_settings) for i in range(R - 1, -1, -1)] + [last_block])
        if global_conds != 0:
            self.class_emb = nn.Linear(global_conds, nf(0), False)
            if spectral:
                self.class_emb = spectral_norm(self.class_emb)
        else:
            self.class_emb = None
        self.linear = GeneralConv(nf(0), 1, kernel_size=1, equalized=equalized, act_alpha=-1, spectral=spectral,
                                  init=init)
        self.depth = 0
        self.alpha = 1.0
        self.max_depth = len(self.blocks) - 1
        self.output_sizes = [initial_kernel_size]
        for u, d in zip(self.progression_scale_up, self.progression_scale_down):
            self.output_sizes.append(int(self.output_sizes[-1] * u / d))
        self.temporal_condition_emb = nn.ModuleDict()
        for k, v in temporal_conds.items():
            module_list = nn.ModuleList(
                [GeneralConv(v, nf(i + 1), kernel_size=1, equalized=False, act_alpha=-1, spectral=spectral, init=init)
                 for i in range(R - 1, -2, -1)])
            self.temporal_condition_emb[k] = module_list

    def find_appropriate_layer(self, cond):
        cond_length = cond.shape[2]
        for i, o in enumerate(self.output_sizes):
            if cond_length <= o:
                return i
            if i == self.depth:
                return i
        assert False, 'we should not reach here'

    def forward(self, x):
        if isinstance(x, dict):
            x, y = x['x'], x
        else:
            y = None
        layer_to_conds = {}
        if y is not None:
            for k, v in y.items():
                if k.startswith('temporal_'):
                    l = len(self.blocks) - self.find_appropriate_layer(v) - 1
                    if l in layer_to_conds:
                        layer_to_conds[l].append(k)
                    else:
                        layer_to_conds[l] = [k]
        cond_loss = 0
        xhighres = x
        h = self.blocks[-(self.depth + 1)](xhighres, True)
        current_layer = len(self.blocks) - self.depth - 1
        if current_layer in layer_to_conds:
            for cond in layer_to_conds[current_layer]:
                # TODO instead of changing y(class) -> y_emb / change d_this_layer -> y_dim (like super res)
                current_emb = self.temporal_condition_emb[cond][current_layer](y[cond])
                current_emb = resample_signal(current_emb, current_emb.shape[2], h.shape[2], True)
                cond_loss = cond_loss + (current_emb * h).sum(dim=1).mean(dim=1)
        if self.depth > 0:
            h = resample_signal(h, self.progression_scale_up[self.depth - 1],
                                self.progression_scale_down[self.depth - 1], True)
            if self.alpha < 1.0:
                xlowres = resample_signal(xhighres, self.progression_scale_up[self.depth - 1],
                                          self.progression_scale_down[self.depth - 1], True)
                preult_rgb = self.blocks[-self.depth].fromRGB(xlowres)
                h = h * self.alpha + (1.0 - self.alpha) * preult_rgb
        all_attention_maps = {}
        for i in range(self.depth, 0, -1):
            h = self.blocks[-i](h)
            current_layer = len(self.blocks) - i
            if current_layer in layer_to_conds:
                for cond in layer_to_conds[current_layer]:
                    current_emb = self.temporal_condition_emb[cond][current_layer](y[cond])
                    current_emb = resample_signal(current_emb, current_emb.shape[2], h.shape[2], True)
                    cond_loss = cond_loss + (current_emb * h).sum(dim=1).mean(dim=1)
            if i > 1:
                h = resample_signal(h, self.progression_scale_up[i - 2], self.progression_scale_down[i - 2], True)
            if (i - 2) in self.self_attention:
                h, attention_map = self.self_attention[i - 2](h)
                if attention_map is not None:
                    all_attention_maps[i] = attention_map
        print('debug.disc_h_size', h.shape)
        o = self.linear(h).mean(dim=2).squeeze()
        if y is not None:
            y = [v for k, v in y.items() if k.startswith('global_')]
            if len(y) > 0:
                y = torch.cat(y, dim=1)
                if y.dim() >= 3:
                    raise ValueError()
                emb = self.class_emb(y)
                cond_loss = cond_loss + (emb * h.squeeze()).sum(dim=1)
        return o + cond_loss, h, all_attention_maps


if __name__ == '__main__':
    a = Generator(32, 5, 2048, 128, 32, 3, True, [], [4, 2, 2, 3, 4, 5, 3], [1, 1, 1, 2, 3, 4, 2],
                  'kaiming_normal', 0.2, False, False, False, 'normal', 5, act_norm='batch', split_z=False)
    z = {'z': torch.randn(3, 256), 'global_1': torch.randn(3, 2), 'global_2': torch.randn(3, 1),
         'temporal_1': torch.randn(3, 1, 32), 'temporal_2': torch.randn(3, 1, 256)}
    for i in range(8):
        a.depth = i
        print(a(z)[0].shape)

    a = Generator(32, 5, 2048, 128, 32, 3, True, [], [4, 2, 2, 3, 4, 5, 3], [1, 1, 1, 2, 3, 4, 2],
                  'kaiming_normal', 0.2, False, False, False, 'normal', 0, act_norm='pixel', split_z=False)
    z = {'z': torch.randn(3, 256)}
    for i in range(8):
        a.depth = i
        print(a(z)[0].shape)
    a = Discriminator(32, 5, 512, 512, 2, 3, True, [], [4, 2, 2, 3, 4, 5, 3], [1, 1, 1, 2, 3, 4, 2],
                      'kaiming_normal', 0.2, False, False, False, 4, {'temporal_1': 1, 'temporal_2': 2})
    x = {'x': torch.randn(3, 5, 32), 'global_1': torch.randn(3, 4), 'temporal_1': torch.randn(3, 1, 64),
         'temporal_2': torch.randn(3, 2, 512)}
    print(0, a(x)[0].shape)

    x['x'] = torch.randn(3, 5, 32 * 4)  # 128
    a.depth = 1
    print(1, a(x)[0].shape)
    a.alpha = 0.5
    print(0.5, a(x)[0].shape)

    a.alpha = 1.0
    a.depth = 2
    x['x'] = torch.randn(3, 5, 32 * 4 * 2)  # 256
    print(2, a(x)[0].shape)
    a.alpha = 0.5
    print(1.5, a(x)[0].shape)

    a.depth = 3
    x['x'] = torch.randn(3, 5, 32 * 4 * 2 * 2)  # 512
    a.alpha = 0.5
    print(2.5, a(x)[0].shape)
    a.alpha = 1.0
    print(3, a(x)[0].shape)

    a.depth = 4
    x['x'] = torch.randn(3, 5, 32 * 4 * 2 * 3)  # 768
    a.alpha = 0.5
    print(3.5, a(x)[0].shape)
    a.alpha = 1.0
    print(4, a(x)[0].shape)

    a.depth = 5
    x['x'] = torch.randn(3, 5, 32 * 4 * 2 * 4)  # 1024
    a.alpha = 0.5
    print(4.5, a(x)[0].shape)
    a.alpha = 1.0
    print(5, a(x)[0].shape)

    a.depth = 6
    x['x'] = torch.randn(3, 5, 32 * 4 * 2 * 5)  # 1280
    a.alpha = 0.5
    print(5.5, a(x)[0].shape)
    a.alpha = 1.0
    print(6, a(x)[0].shape)

    a.depth = 7
    x['x'] = torch.randn(3, 5, 32 * 4 * 3 * 5)  # 1920
    a.alpha = 0.5
    print(6.5, a(x)[0].shape)
    a.alpha = 1.0
    print(7, a(x)[0].shape)
