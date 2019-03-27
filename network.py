from typing import Optional

import torch
from torch import nn
from torch.nn.utils import spectral_norm

from utils import pixel_norm, resample_signal
from layers import GeneralConv, SelfAttention, MinibatchStddev, ScaledTanh, PassChannelResidual, ConcatResidual


class GBlock(nn.Module):  # TODO add four_convs flag to Generator
    def __init__(self, ch_in, ch_out, num_channels, ksize=3, equalized=True,
                 initial_kernel_size=None, is_residual=False, no_tanh=False, per_channel_noise=False,
                 spectral=False, to_rgb_mode='pggan', init='kaiming_normal', four_convs=False, **layer_settings):
        super().__init__()
        is_first = initial_kernel_size is not None
        if not four_convs:
            self.c1 = GeneralConv(ch_in, ch_out, equalized=equalized, init=init, spectral=spectral,
                                  kernel_size=initial_kernel_size if is_first else ksize,
                                  pad=initial_kernel_size - 1 if is_first else None, **layer_settings)
            self.c2 = GeneralConv(ch_out, ch_out, equalized=equalized, kernel_size=ksize,
                                  spectral=spectral, init=init, **layer_settings)
            if per_channel_noise:
                self.c1_noise_weight = nn.Parameter(torch.zeros(1, ch_out, 1))
                self.c2_noise_weight = nn.Parameter(torch.zeros(1, ch_out, 1))
            else:
                self.c1_noise_weight, self.c2_noise_weight = None, None
        else:
            self.c1 = GeneralConv(ch_in, ch_in // 4, equalized=equalized, init=init, spectral=spectral,
                                  kernel_size=initial_kernel_size if is_first else ksize,
                                  pad=initial_kernel_size - 1 if is_first else None, **layer_settings)
            self.c2 = GeneralConv(ch_in // 4, ch_in // 4, equalized=equalized, kernel_size=ksize,
                                  spectral=spectral, init=init, **layer_settings)
            self.c3 = GeneralConv(ch_in // 4, ch_in // 4, equalized=equalized, kernel_size=ksize,
                                  spectral=spectral, init=init, **layer_settings)
            self.c4 = GeneralConv(ch_in // 4, ch_out, equalized=equalized, kernel_size=ksize,
                                  spectral=spectral, init=init, **layer_settings)
            if per_channel_noise:
                self.c1_noise_weight = nn.Parameter(torch.zeros(1, ch_in // 4, 1))
                self.c2_noise_weight = nn.Parameter(torch.zeros(1, ch_in // 4, 1))
                self.c3_noise_weight = nn.Parameter(torch.zeros(1, ch_in // 4, 1))
                self.c4_noise_weight = nn.Parameter(torch.zeros(1, ch_out, 1))
            else:
                self.c1_noise_weight, self.c2_noise_weight = None, None
                self.c3_noise_weight, self.c4_noise_weight = None, None
        if to_rgb_mode == 'pggan':
            to_rgb = GeneralConv(ch_out, num_channels, kernel_size=1, act_alpha=-1,
                                 equalized=equalized, spectral=spectral, init=init)
        elif to_rgb_mode in {'sngan', 'sagan'}:
            to_rgb = GeneralConv(ch_out, num_channels if to_rgb_mode == 'sngan' else ch_out, kernel_size=3,
                                 act_alpha=0.2, equalized=equalized, spectral=spectral, num_classes=0, init=init)
            if to_rgb_mode == 'sagan':
                to_rgb = nn.Sequential(
                    GeneralConv(ch_out, num_channels, kernel_size=1, act_alpha=-1, equalized=equalized,
                                spectral=spectral, init=init), to_rgb)
        elif to_rgb_mode == 'biggan':
            to_rgb = nn.Sequential(nn.BatchNorm1d(ch_out), nn.ReLU(),
                                   GeneralConv(ch_out, num_channels, kernel_size=3, act_alpha=-1, equalized=equalized,
                                               spectral=False, init=init))
        else:
            raise ValueError()
        if no_tanh:
            self.toRGB = to_rgb
        else:
            self.toRGB = nn.Sequential(to_rgb, ScaledTanh())
        if not four_convs:
            if not is_first and is_residual:
                self.residual = nn.Sequential() if ch_in == ch_out else \
                    GeneralConv(ch_in, ch_out, 1, equalized, init=init, act_alpha=-1, spectral=spectral)
            else:
                self.residual = None
        else:
            self.residual = PassChannelResidual()
        self.four_conv = four_convs

    @staticmethod
    def get_per_channel_noise(noise_weight):
        return None if noise_weight is None else torch.randn(*noise_weight.size()) * noise_weight

    def forward(self, x, y=None, last=False):
        if not self.four_conv:
            h = self.c1(x, y=y, z=self.get_per_channel_noise(self.c1_noise_weight))
            h = self.c2(h, y=y, z=self.get_per_channel_noise(self.c2_noise_weight))
        else:
            h = self.c1(x, y=y, z=self.get_per_channel_noise(self.c1_noise_weight))
            h = self.c2(h, y=y, z=self.get_per_channel_noise(self.c2_noise_weight))
            h = self.c3(h, y=y, z=self.get_per_channel_noise(self.c3_noise_weight))
            h = self.c4(h, y=y, z=self.get_per_channel_noise(self.c4_noise_weight))
        if not self.four_conv:
            if self.residual is not None:
                h = h + self.residual(x)
        else:
            h = self.residual(h, x)
        if last:
            return self.toRGB(h)
        return h


class Generator(nn.Module):
    def __init__(self, initial_kernel_size, num_channels, fmap_base, fmap_max, fmap_min, kernel_size, equalized,
                 self_attention_layers, progression_scale_up, progression_scale_down, init, act_alpha,
                 residual, sagan_non_local, factorized_attention, z_distribution, num_conds, to_rgb_mode: str = 'pggan',
                 latent_size: int = 256, normalize_latents: bool = True, dropout: float = 0.2, spectral: bool = False,
                 act_norm: Optional[str] = 'pixel', no_tanh: bool = False, per_channel_noise=False, split_z=False,
                 embed_classes_size: int = 64):
        # num_conds = num_global_conditions + sum([temporal_conditions])
        super().__init__()
        R = len(progression_scale_up)
        self.progression_scale_up = progression_scale_up
        self.progression_scale_down = progression_scale_down

        def nf(stage):
            return min(max(int(fmap_base / (2.0 ** stage)), fmap_min), fmap_max)

        if latent_size is None:
            latent_size = nf(0)
        self.input_latent_size = latent_size
        if num_conds != 0:
            self.y_encoder = GeneralConv(num_conds, embed_classes_size, kernel_size=1, equalized=False,
                                         act_alpha=act_alpha, spectral=False, bias=False)
        else:
            self.y_encoder = None
        num_conds = num_conds if num_conds == 0 else embed_classes_size
        if split_z:
            latent_size //= R + 2  # we also give part of the z to the first layer
            num_conds += latent_size
        self.normalize_latents = normalize_latents
        layer_settings = dict(do=dropout, num_classes=num_conds, act_norm=act_norm, act_alpha=act_alpha)
        self.block0 = GBlock(latent_size, nf(1), num_channels, ksize=kernel_size, equalized=equalized,
                             initial_kernel_size=initial_kernel_size, is_residual=residual, spectral=spectral,
                             no_tanh=no_tanh, to_rgb_mode=to_rgb_mode, init=init, per_channel_noise=per_channel_noise,
                             **layer_settings)
        dummy = []  # to make SA layers registered
        self.self_attention = dict()
        for layer in self_attention_layers:
            dummy.append(SelfAttention(nf(layer + 1), sagan_non_local, spectral, factorized_attention, init))
            self.self_attention[layer] = dummy[-1]
        self.dummy = nn.ModuleList(dummy)
        self.blocks = nn.ModuleList([GBlock(nf(i + 1), nf(i + 2), num_channels,
                                            ksize=kernel_size, equalized=equalized, is_residual=residual,
                                            spectral=spectral, no_tanh=no_tanh, to_rgb_mode=to_rgb_mode,
                                            init=init, per_channel_noise=per_channel_noise, **layer_settings)
                                     for i in range(R)])
        self.depth = 0
        self.alpha = 1.0
        self.split_z = split_z
        self.latent_size = latent_size
        self.max_depth = len(self.blocks)
        self.z_distribution = z_distribution

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

    def forward(self, z):
        if isinstance(z, dict):
            z, y = z['z'], z
        else:
            y = None
        if y is not None:
            saved_inputs = y
            concatenated_y = []
            max_t = 1
            for k, v in y.items():
                if k.startswith('temporal_'):
                    max_t = max(max_t, v.shape[2])
            for k, v in y.items():
                if k.startswith('global_'):
                    concatenated_y.append(v.unsqueeze(-1).expand(-1, -1, max_t))
                elif k.startswith('temporal_'):
                    concatenated_y.append(resample_signal(v, v.shape[2], max_t, pytorch=True))
            if len(concatenated_y) > 0:
                y = torch.cat(concatenated_y, dim=1)
            else:
                y = None
        else:
            saved_inputs = {'z': z}
        if self.normalize_latents:
            z = pixel_norm(z)
        if y is not None and self.y_encoder is not None:
            y = self.y_encoder(y)
        h = z.unsqueeze(2)
        if self.split_z:
            h = h[:, :self.latent_size, :]
        h = self.block0(h, self._cat_z(-1, y, z), self.depth == 0)
        if self.depth == 0:
            saved_inputs['x'] = h
            return saved_inputs, {}
        all_attention_maps = {}
        for i in range(self.depth - 1):
            h, attention_map = self.do_layer(i, h, y, z)
            if attention_map is not None:
                all_attention_maps[i] = attention_map
        h = resample_signal(h, self.progression_scale_down[self.depth - 1], self.progression_scale_up[self.depth - 1],
                            True)
        ult = self.blocks[self.depth - 1](h, self._cat_z(self.depth - 1, y, z), True)
        if self.alpha == 1.0:
            saved_inputs['x'] = ult
            return saved_inputs, all_attention_maps
        preult_rgb = self.blocks[self.depth - 2].toRGB(h) if self.depth > 1 else self.block0.toRGB(h)
        saved_inputs['x'] = preult_rgb * (1.0 - self.alpha) + ult * self.alpha
        return saved_inputs, all_attention_maps


class DBlock(nn.Module):  # TODO add four_convs flag to Discriminator
    def __init__(self, ch_in, ch_out, num_channels, initial_kernel_size=None, is_residual=False,
                 ksize=3, equalized=True, group_size=4, act_alpha: float = 0.0, spectral=False, sngan_rgb=False,
                 init='kaiming_normal', four_convs=False, **layer_settings):
        super().__init__()
        is_last = initial_kernel_size is not None
        self.fromRGB = GeneralConv(num_channels, ch_in, kernel_size=1, equalized=equalized,
                                   act_alpha=-1 if sngan_rgb else act_alpha, spectral=spectral, init=init)
        self.net = []
        if is_last:
            self.net.append(MinibatchStddev(group_size))
        self.net.append(
            GeneralConv(ch_in + (1 if is_last else 0), ch_in if not four_convs else ch_out // 4, kernel_size=ksize,
                        equalized=equalized, act_alpha=act_alpha, spectral=spectral, init=init, **layer_settings))
        self.net.append(GeneralConv(ch_in if not four_convs else ch_out // 4, ch_out if not four_convs else ch_out // 4,
                                    kernel_size=initial_kernel_size if is_last else ksize, pad=0 if is_last else None,
                                    init=init, equalized=equalized, act_alpha=act_alpha, spectral=spectral,
                                    **layer_settings))
        if four_convs:
            self.net.append(
                GeneralConv(ch_out // 4, ch_out // 4, kernel_size=ksize, equalized=equalized, act_alpha=act_alpha,
                            spectral=spectral, init=init, **layer_settings))
            self.net.append(
                GeneralConv(ch_out // 4, ch_out, kernel_size=ksize, equalized=equalized, act_alpha=act_alpha,
                            spectral=spectral, init=init, **layer_settings))
        self.net = nn.Sequential(*self.net)
        self.is_last = initial_kernel_size
        if not four_convs:
            if is_residual and not is_last:
                self.residual = nn.Sequential() if ch_in == ch_out else GeneralConv(ch_in, ch_out, kernel_size=1,
                                                                                    equalized=equalized, act_alpha=-1,
                                                                                    spectral=spectral, init=init)
            else:
                self.residual = None
        else:
            self.residual = ConcatResidual(ch_in, ch_out, equalized, spectral, init)
        self.four_convs = four_convs

    def forward(self, x, first=False):
        if first:
            x = self.fromRGB(x)
        h = self.net(x)
        if self.four_convs:
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
