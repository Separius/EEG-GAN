import torch
import torch.nn as nn
import numpy as np
from functools import partial
from network import NeoPGConv1d, ToRGB, pixel_norm
from utils import cudize


class Cnn2Rnn(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.permute(2, 0, 1)


class DRNN(nn.Module):

    def __init__(self, depth, n_input, n_hiddens, n_layers, dropout=0, cell_type='GRU', bidir=False, dilation=2,
                 from_conv=nn.Conv1d):
        super(DRNN, self).__init__()
        self.dilation = dilation
        self.dilations = [dilation ** i for i in range(depth)]
        self.is_lstm = cell_type.lower() == 'lstm'
        self.depth = 0
        self.alpha = 1
        self.bi = bidir
        self.n_layers = n_layers
        cell = {'gru': nn.GRU, 'rnn': nn.RNN, 'lstm': nn.LSTM}[cell_type.lower()]
        self.cells = nn.ModuleList([cell(n_input if i == 0 else (n_hiddens[i - 1] * (2 if bidir else 1)), n_hiddens[i],
                                         num_layers=n_layers, dropout=dropout, bidirectional=bidir) for i in
                                    range(depth)])
        self.froms = nn.ModuleList(
            [nn.Sequential(from_conv(n_input, n_input if i == 0 else (n_hiddens[i - 1] * (2 if bidir else 1)), 1),
                           Cnn2Rnn()) for i in
             range(depth)])

    def forward(self, inputs):
        second_layer_inputs = self.froms[-self.depth](inputs) if self.alpha != 1 else None
        inputs = self.froms[-self.depth - 1](inputs)
        for i, (cell, dilation) in enumerate(zip(self.cells[-self.depth - 1:], self.dilations)):
            inputs = self.drnn_layer(cell, inputs, dilation, second_layer_inputs if i == 1 else None)
        return inputs.permute(1, 2, 0)

    def drnn_layer(self, cell, inputs, rate, second_layer_inputs=None):
        batch_size = inputs[0].size(0)
        hidden_size = cell.hidden_size
        dilated_inputs = self._prepare_inputs(inputs, rate)
        if second_layer_inputs is not None:
            dilated_inputs = self.alpha * dilated_inputs + (1.0 - self.alpha) * self._prepare_inputs(
                second_layer_inputs, rate)
        dilated_outputs = self._apply_cell(dilated_inputs, cell, batch_size, rate, hidden_size)
        splitted_outputs = self._split_outputs(dilated_outputs, rate)
        return splitted_outputs

    @staticmethod
    def _apply_cell(dilated_inputs, cell, batch_size, rate, hidden_size):
        dilated_outputs, _ = cell(dilated_inputs)
        return dilated_outputs

    @staticmethod
    def _split_outputs(dilated_outputs, rate):
        batchsize = dilated_outputs.size(1) // rate
        blocks = [dilated_outputs[:, i * batchsize: (i + 1) * batchsize, :] for i in range(rate)]
        interleaved = torch.stack(blocks).transpose(1, 0).contiguous()
        interleaved = interleaved.view(dilated_outputs.size(0) * rate, batchsize, dilated_outputs.size(2))
        return interleaved

    @staticmethod
    def _prepare_inputs(inputs, rate):
        dilated_inputs = torch.cat([inputs[j::rate, :, :] for j in range(rate)], 1)
        return dilated_inputs


def test():
    drnn = DRNN(depth=5, n_input=3, n_hiddens=[11, 10, 9, 8, 7], n_layers=2, dropout=0.2, bidir=True, dilation=4)
    bs = 5
    for d in range(5):
        inp = torch.randn(bs, 3, 4 * (4 ** d))
        drnn.depth = d
        drnn.alpha = 0.5
        print(drnn(inp).size())
        drnn.alpha = 1
        print(drnn(inp).size())


class DilatedGenerator(nn.Module):
    def __init__(self, progression_scale, dataset_shape, initial_size, fmap_base, fmap_max, fmap_min, equalized,
                 latent_size=256, normalize_latents=True, pixelnorm=True, activation='lrelu', dropout=0.1,
                 do_mode='mul', spectral_norm=False, ch_by_ch=False, normalization=None, recurrent_dropout=0.1,
                 bidir=False, n_layers=2, cell_type='gru'):
        super(DilatedGenerator, self).__init__()
        resolution = dataset_shape[-1]
        num_channels = dataset_shape[1]
        R = int(np.log2(resolution) / np.log2(progression_scale))
        self.R = R

        def nf(stage):
            return min(max(int(fmap_base / (2.0 ** stage)), fmap_min), fmap_max)

        if latent_size is None:
            latent_size = nf(initial_size - 2)
        self.normalize_latents = normalize_latents
        self.progression_scale = progression_scale
        layer_settings = dict(equalized=equalized, pixelnorm=pixelnorm, act=activation, do=dropout, do_mode=do_mode,
                              spectral=spectral_norm, normalization=normalization)
        self.drnn = DRNN(R - initial_size + 1, latent_size, [nf(i) for i in range(R - initial_size + 1)], n_layers,
                         dropout=recurrent_dropout, cell_type=cell_type, bidir=bidir, dilation=progression_scale,
                         from_conv=partial(NeoPGConv1d, **layer_settings))
        self._depth = 0
        self._alpha = 1.0
        self.latent_size = latent_size
        self.max_depth = len(self.drnn.cells) - 1
        self.toRGB = ToRGB(nf(R - initial_size) * (2 if bidir else 1), num_channels,
                           normalization=None if normalization == 'batch_norm' else normalization, ch_by_ch=ch_by_ch,
                           equalized=equalized)

    def set_gamma(self, new_gamma):
        pass

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        self.drnn.alpha = value
        self._alpha = value

    @property
    def depth(self):
        return self._depth

    @depth.setter
    def depth(self, value):
        self.drnn.depth = value
        self._depth = value

    def forward(self, x):
        h = cudize(torch.randn(x.size(0), self.latent_size, self.progression_scale ** self.depth))
        if self.normalize_latents:
            h = pixel_norm(h)
        h = self.drnn(h)
        return self.toRGB(h)


class DilatedDiscriminator(nn.Module):
    def __init__(self, progression_scale, dataset_shape, initial_size, fmap_base, fmap_max, fmap_min, equalized,
                 activation='lrelu', dropout=0.1, do_mode='mul', spectral_norm=False, normalization=None,
                 recurrent_dropout=0.1, bidir=False, n_layers=2, cell_type='gru'):
        super(DilatedDiscriminator, self).__init__()
        resolution = dataset_shape[-1]
        num_channels = dataset_shape[1]
        R = int(np.log2(resolution) / np.log2(progression_scale))
        self.R = R

        def nf(stage):
            return min(max(int(fmap_base / (2.0 ** stage)), fmap_min), fmap_max)

        layer_settings = dict(act=activation, do=dropout, do_mode=do_mode, equalized=equalized,
                              spectral_norm=spectral_norm, normalization=normalization)
        self.drnn = DRNN(R - initial_size + 1, num_channels, [nf(i) for i in range(R - initial_size + 1)], n_layers,
                         dropout=recurrent_dropout, cell_type=cell_type, bidir=bidir, dilation=progression_scale,
                         from_conv=nn.Conv1d)
        self.toPred = NeoPGConv1d(nf(R - initial_size) * (2 if bidir else 1), 1, ksize=1,
                                  normalization=None if normalization == 'batch_norm' else normalization,
                                  equalized=equalized, spectral=spectral_norm, act=None)
        self._depth = 0
        self._alpha = 1.0
        self.max_depth = len(self.drnn.cells) - 1

    def set_gamma(self, new_gamma):
        pass

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        self.drnn.alpha = value
        self._alpha = value

    @property
    def depth(self):
        return self._depth

    @depth.setter
    def depth(self, value):
        self.drnn.depth = value
        self._depth = value

    def forward(self, x):
        h = self.drnn(x)
        return self.toPred(h).mean(dim=2), h
