import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from utils import cudize
from torch.nn import Parameter
from torch.nn.init import calculate_gain


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


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


class EqualizedConv1d(nn.Module):
    def __init__(self, c_in, c_out, k_size, stride=1, padding=0):
        super(EqualizedConv1d, self).__init__()
        self.conv = nn.Conv1d(c_in, c_out, k_size, stride, padding, bias=False)
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


class Cnn2RnnConverter(nn.Module):
    def __init__(self):
        super(Cnn2RnnConverter, self).__init__()

    def forward(self, input):
        return cnn2rnn(input)


def cnn2rnn(x):
    return x.permute(2, 0, 1)


def rnn2cnn(x):
    return x.permute(1, 2, 0)


cell_dict = {'rnn': torch.nn.RNN, 'gru': torch.nn.GRU, 'lstm': torch.nn.LSTM}


def get_cell(name):
    return cell_dict[name.lower()]


class Rnn2CnnConverter(nn.Module):
    def __init__(self):
        super(Rnn2CnnConverter, self).__init__()

    def forward(self, input):
        return rnn2cnn(input[0])


def get_recurrent_layer(cell_type, ch_in, ch_out):
    return nn.Sequential(Cnn2RnnConverter(), get_cell(cell_type)(ch_in, ch_out, ), Rnn2CnnConverter())


class ChannelByChannelOut(nn.Module):
    def __init__(self, ch_in, ch_out, equalized=True):
        super(ChannelByChannelOut, self).__init__()
        self.num_out = ch_out
        self.out = nn.ModuleList([nn.Sequential(
            NeoPGConv1d(ch_in + i, 1, ksize=1, pixelnorm=False, act=None, equalized=equalized), ScaledTanh()) for i in
            range(ch_out)])

    def forward(self, x):
        r = x
        for o in self.out:
            r = torch.cat([r, o(r)], dim=1)
        return r[:, -self.num_out:, :]


class ToRGB(nn.Module):
    def __init__(self, ch_in, num_channels, ch_by_ch=False, recurrent=None, equalized=True):
        super(ToRGB, self).__init__()
        self.num_channels = num_channels
        if recurrent is None:
            self.rnn = None
        else:
            self.rnn = get_cell(recurrent)(ch_in + num_channels, hidden_size=ch_in // 4)
        if ch_by_ch:
            self.toRGB = ChannelByChannelOut(ch_in // 4 if recurrent else ch_in, num_channels, equalized=equalized)
        else:
            self.toRGB = nn.Sequential(
                NeoPGConv1d(ch_in // 4 if recurrent else ch_in, num_channels, ksize=1, pixelnorm=False, act=None,
                            equalized=equalized), ScaledTanh())

    def forward(self, x):
        if self.rnn is None:
            return self.toRGB(x)
        ret = cudize(torch.zeros(x.size(0), self.num_channels, 1))
        ans = []
        state = None
        for t in range(x.size(2)):
            ret, state = self.rnn(cnn2rnn(torch.cat((x[:, :, t:t + 1], ret), dim=1)), state)
            ret = self.toRGB(rnn2cnn(ret))
            ans.append(ret)
        return torch.cat(ans, dim=2)


class NeoPGConv1d(nn.Module):
    def __init__(self, ch_in, ch_out, ksize=3, equalized=True, pad=None, pixelnorm=True,
                 act='lrelu', do=0, do_mode='mul', spectral=False, phase_shuffle=0):
        super(NeoPGConv1d, self).__init__()
        pad = (ksize - 1) // 2 if pad is None else pad
        conv = EqualizedConv1d if equalized else nn.Conv1d
        conv = conv(ch_in, ch_out, ksize, 1, pad)
        self.net = [SpectralNorm(conv) if spectral else conv]
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


class GBlock(nn.Module):  # TODO work on the recurrent mode
    def __init__(self, ch_in, ch_out, num_channels, ksize=3, equalized=True, initial_size=None, ch_by_ch=False,
                 recurrent_to_rgb=None, layer_recurrent=None, **layer_settings):
        super(GBlock, self).__init__()
        c1 = NeoPGConv1d(ch_in, ch_out, equalized=equalized, ksize=2 ** initial_size if initial_size else ksize,
                         pad=2 ** initial_size - 1 if initial_size else None, **layer_settings)
        if layer_recurrent is None:
            c2 = NeoPGConv1d(ch_out, ch_out, equalized=equalized, **layer_settings)
        else:
            c2 = get_recurrent_layer(layer_recurrent, ch_out, ch_out)
        self.net = nn.Sequential(c1, c2)
        self.toRGB = ToRGB(ch_out, num_channels, ch_by_ch=ch_by_ch, recurrent=recurrent_to_rgb, equalized=equalized)

    def forward(self, x, last=False):
        x = self.net(x)
        return self.toRGB(x) if last else x


class Generator(nn.Module):
    def __init__(self, dataset_shape, initial_size=2, fmap_base=2048, fmap_max=256, fmap_min=16, latent_size=256,
                 upsample='nearest', normalize_latents=True, pixelnorm=True, activation='lrelu', dropout=0,
                 do_mode='mul', equalized=True, spectral_norm=False, ch_by_ch=False, kernel_size=3,
                 recurrent_to_rgb=None, layer_recurrent=None, phase_shuffle=0):
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
        layer_settings = dict(pixelnorm=pixelnorm, act=activation, do=dropout, do_mode=do_mode, spectral=spectral_norm,
                              phase_shuffle=phase_shuffle)
        self.block0 = GBlock(latent_size, nf(1), num_channels, ksize=kernel_size, equalized=equalized,
                             initial_size=initial_size, ch_by_ch=ch_by_ch, recurrent_to_rgb=recurrent_to_rgb,
                             layer_recurrent=layer_recurrent, **layer_settings)
        self.blocks = nn.ModuleList([GBlock(nf(i - 1), nf(i), num_channels, ksize=kernel_size, equalized=equalized,
                                            ch_by_ch=ch_by_ch, recurrent_to_rgb=recurrent_to_rgb,
                                            layer_recurrent=layer_recurrent, **layer_settings) for i in
                                     range(initial_size, R)])
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


class FromRGB(nn.Module):
    def __init__(self, num_channels, ch_out, recurrent=None, equalized=True, spectral=False):
        super(FromRGB, self).__init__()
        self.num_channels = num_channels
        if recurrent is None:
            self.rnn = None
        else:
            self.rnn = get_recurrent_layer(recurrent, num_channels, ch_out // 4)
        self.fromRGB = NeoPGConv1d(num_channels if recurrent is None else ch_out // 4, ch_out, ksize=1, pixelnorm=False,
                                   equalized=equalized, spectral=spectral)

    def forward(self, x):
        if self.rnn is None:
            return self.fromRGB(x)
        return self.fromRGB(self.rnn(x))


class DBlock(nn.Module):
    def __init__(self, ch_in, ch_out, num_channels, ksize=3, from_recurrent=None, layer_recurrent=None,
                 equalized=True, spectral=False, **layer_settings):
        super(DBlock, self).__init__()
        self.fromRGB = FromRGB(num_channels, ch_in, recurrent=from_recurrent, equalized=equalized, spectral=spectral)
        c1 = NeoPGConv1d(ch_in, ch_in, ksize=ksize, equalized=equalized, **layer_settings)
        if layer_recurrent is None:
            c2 = NeoPGConv1d(ch_in, ch_out, ksize=ksize, equalized=equalized, **layer_settings)
        else:
            c2 = get_recurrent_layer(layer_recurrent, ch_in, ch_out)
        self.net = nn.Sequential(c1, c2)

    def forward(self, x, first=False):
        if first:
            x = self.fromRGB(x)
        return self.net(x)


class DLastBlock(nn.Module):
    def __init__(self, ch_in, ch_out, num_channels, initial_size=2, temporal=False, num_stat_channels=1, ksize=3,
                 from_recurrent=None, layer_recurrent=None, equalized=True, spectral=False, **layer_settings):
        super(DLastBlock, self).__init__()
        self.fromRGB = FromRGB(num_channels, ch_in, recurrent=from_recurrent, equalized=equalized, spectral=spectral)
        self.net = [MinibatchStddev(temporal, num_stat_channels)]
        if layer_recurrent is None:
            self.net.append(
                NeoPGConv1d(ch_in + num_stat_channels, ch_in, ksize=ksize, equalized=equalized, **layer_settings))
        else:
            self.net.append(get_recurrent_layer(layer_recurrent, ch_in + num_stat_channels, ch_in))
        self.net.append(
            NeoPGConv1d(ch_in, ch_out, ksize=2 ** initial_size, pad=0, equalized=equalized, **layer_settings))
        self.net = nn.Sequential(*self.net)

    def forward(self, x, first=False):
        if first:
            x = self.fromRGB(x)
        return self.net(x)


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
    def __init__(self, dataset_shape, apply_sigmoid, initial_size=2, fmap_base=2048, fmap_max=256, fmap_min=64,
                 downsample='average', pixelnorm=False, activation='lrelu', dropout=0, do_mode='mul', equalized=True,
                 spectral_norm=False, kernel_size=3, recurrent_from_rgb=None, layer_recurrent=None, phase_shuffle=0,
                 temporal_stats=False, num_stat_channels=1):
        super(Discriminator, self).__init__()
        resolution = dataset_shape[-1]
        num_channels = dataset_shape[1]
        R = int(np.log2(resolution))
        assert resolution == 2 ** R and resolution >= initial_size ** 2
        self.R = R

        def nf(stage):
            return max(min(max(int(fmap_base / (2.0 ** stage)), fmap_min), fmap_max), 2)

        layer_settings = dict(pixelnorm=pixelnorm, act=activation, do=dropout, do_mode=do_mode, spectral=spectral_norm,
                              phase_shuffle=phase_shuffle)
        last_block = DLastBlock(nf(initial_size - 1), nf(initial_size - 2), num_channels, initial_size=initial_size,
                                temporal=temporal_stats, num_stat_channels=num_stat_channels, ksize=kernel_size,
                                from_recurrent=recurrent_from_rgb, layer_recurrent=layer_recurrent, equalized=equalized,
                                spectral=spectral_norm, **layer_settings)
        self.blocks = nn.ModuleList([DBlock(nf(i), nf(i - 1), num_channels, ksize=kernel_size,
                                            from_recurrent=recurrent_from_rgb, layer_recurrent=layer_recurrent,
                                            equalized=equalized, spectral=spectral_norm, **layer_settings) for i in
                                     range(R - 1, initial_size - 1, -1)] + [last_block])
        self.linear = nn.Linear(nf(initial_size - 2), 1)
        if spectral_norm:
            self.linear = SpectralNorm(self.linear)
        if apply_sigmoid:
            self.linear = nn.Sequential(self.linear, nn.Sigmoid())
        self.depth = 0
        self.alpha = 1.0
        self.max_depth = len(self.blocks) - 1
        if downsample == 'average':
            self.downsampler = nn.AvgPool1d(kernel_size=2)
        elif downsample == 'stride':
            self.downsampler = DownSample(scale_factor=2)

    def set_depth(self, depth):
        self.depth = depth

    def set_alpha(self, alpha):
        self.alpha = alpha

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
