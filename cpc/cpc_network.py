import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from pytorch_pretrained_bert.modeling import BertLayer

from utils import cudize
from cpc.cpc_loss import KPredLoss, OneOneMI, SeqOneMI


class SincEncoder(nn.Module):
    def __init__(self, num_channels, is_shared=True, kernel_size=121,
                 num_kernels=16, sample_rate=60.0, min_low_hz=0.0, min_band_hz=1.0):
        super().__init__()
        self.is_shared = is_shared
        if is_shared:
            self.sinc = SincConv(kernel_size, num_kernels, sample_rate, min_low_hz, min_band_hz)
        else:
            self.sinc = nn.ModuleList([SincConv(kernel_size, num_kernels) for i in range(num_channels)])

    def forward(self, x):
        B, C, T = x.shape
        if self.is_shared:
            return self.sinc(x.view(B * C, 1, T)).view(B, -1, T)
        return torch.cat([self.sinc[i](x[:, i:i + 1, :]) for i in range(C)], dim=1)


class SincConv(nn.Module):
    def __init__(self, kernel_size, out_channels, sample_rate=60.0, min_low_hz=0.0, min_band_hz=1.0):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        if kernel_size % 2 == 0:  # Forcing the filters to be odd (i.e, perfectly symmetrics)
            self.kernel_size = self.kernel_size + 1
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        hz = np.linspace(min_low_hz, sample_rate / 2 - (min_low_hz + min_band_hz), out_channels + 1)
        # filter lower frequency (out_channels, 1)
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))
        # filter frequency band (out_channels, 1)
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))
        # Hamming window
        n_lin = torch.linspace(0, (self.kernel_size / 2) - 1, steps=int((self.kernel_size / 2)))
        self.window_ = 0.54 - 0.46 * torch.cos(2.0 * math.pi * n_lin / self.kernel_size)
        n = (self.kernel_size - 1) / 2.0
        self.n_ = 2 * math.pi * torch.arange(-n, 0).view(1, -1) / self.sample_rate

    def forward(self, waveforms):
        self.n_ = self.n_.to(waveforms.device)
        self.window_ = self.window_.to(waveforms.device)
        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_), self.min_low_hz, self.sample_rate / 2)
        band = (high - low)[:, 0]
        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)
        band_pass_left = ((torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (self.n_ / 2)) * self.window_
        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])
        band_pass = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=1)
        band_pass = band_pass / (2 * band[:, None])
        self.filters = band_pass.view(self.out_channels, 1, self.kernel_size)
        x_p = F.pad(waveforms, (self.kernel_size // 2, self.kernel_size // 2), mode='reflect')
        return F.conv1d(x_p, self.filters, bias=None)


class PriorDiscriminator(nn.Module):
    def __init__(self, latent_size, prior='uniform'):
        super().__init__()
        prior = prior.lower()
        assert prior in {'uniform', 'gaussian'}
        self.latent_size = latent_size
        self.prior = prior
        self.network = nn.Sequential(spectral_norm(nn.Linear(latent_size, latent_size * 8)), nn.ReLU(),
                                     spectral_norm(nn.Linear(latent_size * 8, latent_size * 2)), nn.ReLU(),
                                     spectral_norm(nn.Linear(latent_size * 2, 1)))

    def forward(self, x):
        assert x.size(1) == self.latent_size
        x = x.view(-1, self.latent_size)
        return self.network(x)


class ConvEncoder(nn.Module):
    def __init__(self, input_channels=5, long=True, use_pooling=False, activation='relu',
                 dropout=0.1, use_sinc=False, shared_sinc=True, tiny=False):
        super().__init__()
        activation = activation.lower()
        if long:  # generates 32 codes of size 128
            down_ratios = [5, 4, 3]
            channel_sizes = [32, 64, 128]
        else:  # generates 16 codes of size 128
            down_ratios = [5, 4, 3, 2]
            channel_sizes = [32, 64, 128, 128]
        if tiny:
            channel_sizes = [cs // 2 for cs in channel_sizes]
        kernel_sizes = [2 * dr - 1 for dr in down_ratios]
        self.z_size = channel_sizes[-1]
        is_glu = False
        activation = activation.lower()
        if activation == 'relu':
            act = nn.ReLU()
        elif activation == 'lrelu':
            act = nn.LeakyReLU(0.2)
        elif activation == 'selu':
            act = nn.SELU()
        elif activation == 'glu':
            act = nn.GLU(dim=1)
            is_glu = True
        else:
            raise ValueError('invalid activation')
        net = []
        if use_sinc:
            net.append(SincEncoder(input_channels, shared_sinc))
            multiplier = 2 if shared_sinc else 1
            net.append(nn.Conv1d(input_channels * multiplier * 8, input_channels * multiplier * 2, 1))
            net.append(nn.ReLU())
            input_channels = input_channels * multiplier * 2
        last_layer = len(down_ratios) - 1
        for i, (dr, cs, ks) in enumerate(zip(down_ratios, channel_sizes, kernel_sizes)):
            net.append(nn.Conv1d(input_channels, cs * (2 if is_glu else 1), ks,
                                 stride=1 if use_pooling else dr, padding=(ks - 1) // 2))
            net.append(act)
            if dropout != 0.0 and i != last_layer:
                net.append(nn.Dropout(dropout))
            if use_pooling:
                net.append(nn.MaxPool1d(dr))
            if i != last_layer:
                net.append(nn.BatchNorm1d(cs))
            input_channels = cs
        self.network = nn.Sequential(*net)

    def forward(self, x):
        return self.network(x)


class PNormPooling(nn.Module):
    def __init__(self, input_size, batch_norm=True, mlp_sizes=None, p=None):
        super().__init__()
        if p is None:
            p = [1.0, float('+inf')]
        if mlp_sizes is None:
            mlp_sizes = [input_size * 2, input_size]
        self.p = p
        network = []
        mlp_sizes = [input_size * len(p)] + mlp_sizes
        last_layer = len(mlp_sizes) - 2
        self.pool_size = mlp_sizes[-1]
        for enu, (i, o) in enumerate(zip(mlp_sizes, mlp_sizes[1:])):
            network.append(nn.Linear(i, o))
            if enu != last_layer:
                network.append(nn.ReLU())
                if batch_norm:
                    network.append(nn.BatchNorm1d(o))
        self.network = nn.Sequential(*network)

    def forward(self, x):
        return self.network(torch.cat([torch.norm(x, p, 2) for p in self.p], dim=1))


class AutoRNN(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional=False, cell_type='GRU', num_layers=1, dropout=0):
        super().__init__()
        cell_type = cell_type.lower()
        if cell_type == 'gru':
            cell = nn.GRU
        elif cell_type == 'lstm':
            cell = nn.LSTM
        elif cell_type == 'rnn':
            cell = nn.RNN
        else:
            raise ValueError('invalid cell_type')
        self.forward_rnn = cell(input_size, hidden_size, num_layers, dropout=dropout)
        if bidirectional:
            self.backward_rnn = cell(input_size, hidden_size, num_layers, dropout=dropout)
            self.c_size = hidden_size * 2
        else:
            self.backward_rnn = None
            self.c_size = hidden_size

    def forward(self, x):
        x_permuted = x.permute(2, 0, 1)
        result = self.forward_rnn(x_permuted)[0]
        if self.backward_rnn is not None:
            result = torch.cat([result, self.backward_rnn(x_permuted.flip(0))[0]], dim=2)
        return result.permute(1, 2, 0)


class Transformer(nn.Module):
    class BertConfig:
        def __init__(self, hidden_size, num_heads, dropout):
            self.hidden_size = hidden_size
            self.num_attention_heads = num_heads
            self.attention_probs_dropout_prob = dropout
            self.hidden_dropout_prob = dropout
            self.intermediate_size = hidden_size * 2
            self.hidden_act = 'gelu'  # gelu, relu, swish

    def create_attention_mask(self, seq_len, forward=True):
        if self.causal:
            if forward:
                attention_mask = torch.tril(torch.ones(1, 1, seq_len, seq_len))
            else:
                attention_mask = torch.triu(torch.ones(1, 1, seq_len, seq_len))
        else:
            attention_mask = torch.ones(1, 1, 1, seq_len)
        return (1.0 - attention_mask) * -10000.0

    def __init__(self, input_size, causal=True, bidirectional=False,
                 num_layers=3, num_heads=4, dropout=0.2, max_seq_len=32):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_seq_len, input_size)
        self.causal = causal
        self.bidirectional = bidirectional
        bert_config = self.BertConfig(input_size, num_heads, dropout)
        self.forward_transformer = nn.ModuleList([BertLayer(bert_config) for _ in range(num_layers)])
        self.c_size = input_size
        if bidirectional and causal:
            self.backward_transformer = nn.ModuleList([BertLayer(bert_config) for _ in range(num_layers)])
            self.c_size *= 2
        else:
            self.backward_transformer = None

    def forward(self, x):  # BCT
        pos_embedding = self.pos_embedding(cudize(torch.arange(x.size(2)))).permute(1, 0).unsqueeze(0)
        forward_mask = self.create_attention_mask(x.size(2), True).to(x)
        forward_output = (x + pos_embedding).permute(0, 2, 1)
        for bl in self.forward_transformer:
            forward_output = bl(forward_output, forward_mask)
        forward_output = forward_output.permute(0, 2, 1)
        if self.backward_transformer is None:
            return forward_output
        backward_mask = self.create_attention_mask(x.size(2), False).to(x)
        backward_output = (x + pos_embedding).permute(0, 2, 1)
        for bl in self.backward_transformer:
            backward_output = bl(backward_output, backward_mask)
        backward_output = backward_output.permute(0, 2, 1)
        return torch.cat([forward_output, backward_output], dim=1)


class Network(nn.Module):
    def __init__(self, input_channels, generate_long_sequence=True, pooling=False, encoder_dropout=0.1,
                 use_sinc_encoder=False, use_shared_sinc=True, bidirectional=False, contextualizer_num_layers=1,
                 contextualizer_dropout=0, use_transformer=False, causal_prediction=True, prediction_k=4,
                 encoder_activation='relu', tiny_encoder=True):
        super().__init__()
        encoder = ConvEncoder(input_channels=input_channels, long=generate_long_sequence, use_pooling=pooling,
                              activation=encoder_activation, dropout=encoder_dropout, use_sinc=use_sinc_encoder,
                              shared_sinc=use_shared_sinc, tiny=tiny_encoder)
        z_pooler = PNormPooling(encoder.z_size)
        if use_transformer:
            contextualizer = Transformer(input_size=encoder.z_size, causal=True, bidirectional=bidirectional,
                                         num_heads=4, max_seq_len=32 if generate_long_sequence else 16,
                                         num_layers=contextualizer_num_layers, dropout=contextualizer_dropout)
        else:
            contextualizer = AutoRNN(input_size=encoder.z_size, hidden_size=2 * encoder.z_size,
                                     bidirectional=bidirectional, num_layers=contextualizer_num_layers,
                                     dropout=contextualizer_dropout, cell_type='GRU')
        c_pooler = PNormPooling(contextualizer.c_size)
        prediction_loss_network = KPredLoss(contextualizer.c_size, encoder.z_size, k=prediction_k,
                                            auto_is_bidirectional=bidirectional, look_both=not causal_prediction)
        mi_hidden_size = min(c_pooler.pool_size, z_pooler.pool_size) * 2
        c_pooled_mi_z_pooled = OneOneMI(c_pooler.pool_size, z_pooler.pool_size, hidden_size=mi_hidden_size)
        c_pooled_mi_z = SeqOneMI(c_pooler.pool_size, encoder.z_size, hidden_size=mi_hidden_size)

        self.encoder = encoder
        self.z_pooler = z_pooler
        self.contextualizer = contextualizer
        self.c_pooler = c_pooler
        self.prediction_loss_network = prediction_loss_network
        self.c_pooled_mi_z_pooled = c_pooled_mi_z_pooled
        self.c_pooled_mi_z = c_pooled_mi_z

    def forward(self, x):
        z = self.encoder(x)
        z_pooled = self.z_pooler(z)
        c = self.contextualizer(z)
        c_pooled = self.c_pooler(c)
        prediction_loss, pred_acc = self.prediction_loss_network(c, z)
        global_discriminator_loss, global_accuracy = self.c_pooled_mi_z_pooled(c_pooled, z_pooled)
        local_discriminator_loss, local_accuracy = self.c_pooled_mi_z(c_pooled, z)
        return prediction_loss, global_discriminator_loss, local_discriminator_loss, c_pooled, global_accuracy, local_accuracy, pred_acc

    def inference_forward(self, x):
        z = self.encoder(x)
        z_pooled = self.z_pooler(z)
        c = self.contextualizer(z)
        c_pooled = self.c_pooler(c)
        return z, c, z_pooled, c_pooled

    def complete_forward(self, x):
        z = self.encoder(x)
        z_pooled = self.z_pooler(z)
        c = self.contextualizer(z)
        c_pooled = self.c_pooler(c)
        prediction_loss, pred_acc = self.prediction_loss_network(c, z)
        global_discriminator_loss, global_accuracy = self.c_pooled_mi_z_pooled(c_pooled, z_pooled)
        local_discriminator_loss, local_accuracy = self.c_pooled_mi_z(c_pooled, z)
        return prediction_loss, global_discriminator_loss, local_discriminator_loss, c_pooled, global_accuracy, local_accuracy, pred_acc, z, c, z_pooled
