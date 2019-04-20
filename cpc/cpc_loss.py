import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def generate_prior_samples(prior_discriminator, encoded_input):
    if prior_discriminator.prior == 'uniform':
        return torch.rand_like(encoded_input)
    else:
        return torch.randn_like(encoded_input)


def calc_prior_discriminator_loss(prior_discriminator, encoded_inputs, is_hinge):
    real = prior_discriminator(generate_prior_samples(prior_discriminator, encoded_inputs))
    fake = prior_discriminator(encoded_inputs)
    if is_hinge:
        return loss_hinge_dis(fake, real)
    return loss_dcgan_dis(fake, real)


def calc_prior_encoder_loss(prior_discriminator, encoded_inputs, is_hinge):
    fake = prior_discriminator(encoded_inputs)
    if is_hinge:
        return loss_hinge_gen(fake)
    return loss_dcgan_gen(fake)


def loss_hinge_dis(dis_fake, dis_real):
    return torch.mean(F.relu(1. - dis_real)) + torch.mean(F.relu(1. + dis_fake))


def loss_hinge_gen(dis_fake):
    return -torch.mean(dis_fake)


def loss_dcgan_dis(dis_fake, dis_real):
    return torch.mean(F.softplus(-dis_real)) + torch.mean(F.softplus(dis_fake))


def loss_dcgan_gen(dis_fake):
    return torch.mean(F.softplus(-dis_fake))


def log_sum_exp(x, axis=None):
    x_max = torch.max(x, axis)[0]
    return torch.log((torch.exp(x - x_max)).sum(axis)) + x_max


def get_positive_expectation(p_samples, measure):
    if measure == 'GAN':
        return -F.softplus(-p_samples)
    if measure == 'JSD':
        return math.log(2.) - F.softplus(-p_samples)
    if measure == 'X2':
        return p_samples ** 2
    if measure == 'KL':
        return p_samples + 1.
    if measure == 'RKL':
        return -torch.exp(-p_samples)
    if measure == 'DV':
        return p_samples
    if measure == 'H2':
        return 1. - torch.exp(-p_samples)
    if measure == 'W1':
        return p_samples
    raise ValueError('unsupported measure')


def get_negative_expectation(q_samples, measure):
    if measure == 'GAN':
        return F.softplus(-q_samples) + q_samples
    if measure == 'JSD':
        return F.softplus(-q_samples) + q_samples - math.log(2.)
    if measure == 'X2':
        return -0.5 * ((torch.sqrt(q_samples ** 2) + 1.) ** 2)
    if measure == 'KL':
        return torch.exp(q_samples)
    if measure == 'RKL':
        return q_samples - 1.
    if measure == 'DV':
        return log_sum_exp(q_samples, 0) - math.log(q_samples.size(0))
    if measure == 'H2':
        return torch.exp(q_samples) - 1.
    if measure == 'W1':
        return q_samples
    raise ValueError('unsupported measure')


def calc_one_way_loss(u, positive_mask, mean_dim, measure):
    E_pos = get_positive_expectation(u, measure)
    E_neg = get_negative_expectation(u, measure)
    if mean_dim is not None:
        E_pos = E_pos.mean(mean_dim)
        E_neg = E_neg.mean(mean_dim)
    negative_mask = 1.0 - positive_mask
    E_pos = (E_pos * positive_mask).sum() / positive_mask.sum()
    E_neg = (E_neg * negative_mask).sum() / negative_mask.sum()
    return E_neg - E_pos


def calc_slice_accuracy(u, bs):
    return (u.argmax(dim=0).cpu() == torch.arange(bs)).sum().item() / bs, (
            u.argmax(dim=1).cpu() == torch.arange(bs)).sum().item() / bs


def calc_accuracy(u):
    bs = u.size(0)
    nd = u.ndimension()
    if nd == 2:
        return calc_slice_accuracy(u, bs)
    elif nd == 3:
        t = u.size(2)
        res = [calc_slice_accuracy(u[..., i], bs) for i in range(t)]
        return sum([r[0] for r in res]) / t, sum([r[1] for r in res]) / t


class OneOneMI(nn.Module):
    def __init__(self, code_size, global_size, hidden_size, mode='mlp', measure='JSD'):
        super().__init__()
        measure = measure.upper()
        self.measure = measure
        mode = mode.lower()
        if mode == 'bilinear':
            self.bilinear_weight = nn.Parameter(torch.randn(code_size, global_size))
        elif mode == 'dot':
            assert code_size == global_size
        elif mode == 'mlp' or mode == 'cnn':
            self.network = nn.Sequential(
                nn.Conv1d(global_size + code_size, hidden_size, 1), nn.ReLU(),
                nn.Conv1d(hidden_size, hidden_size, 1), nn.ReLU(),
                nn.Conv1d(hidden_size, 1, 1)
            )
        else:
            raise ValueError('unsupported mode')
        self.mode = mode

    def forward(self, code, g):
        if self.mode == 'bilinear':
            u = torch.einsum('bc,cg,dg->bd', code, self.bilinear_weight, g)
        elif self.mode == 'dot':
            u = torch.einsum('bc,dc->bd', code, g)
        else:
            batch_size = code.size(0)
            code_expanded = code.unsqueeze(0).expand(batch_size, -1, -1)
            g_expanded = g.unsqueeze(1).expand(-1, batch_size, -1)
            u = self.network(torch.cat([code_expanded, g_expanded], dim=2).permute(0, 2, 1)).squeeze()
        positive_mask = torch.eye(u.size(0)).to(u)
        return calc_one_way_loss(u, positive_mask, None, self.measure), calc_accuracy(u)


class SeqOneMI(nn.Module):
    def __init__(self, code_size, seq_size, hidden_size, mode='mlp', measure='JSD'):
        super().__init__()
        measure = measure.upper()
        self.measure = measure
        mode = mode.lower()
        if mode == 'bilinear':
            self.bilinear_weight = nn.Parameter(torch.randn(code_size, seq_size))
        elif mode == 'dot':
            assert code_size == seq_size
        elif mode == 'mlp' or mode == 'cnn':
            self.network = nn.Sequential(
                nn.Conv2d(code_size + seq_size, hidden_size, 1), nn.ReLU(),
                nn.Conv2d(hidden_size, hidden_size, 1), nn.ReLU(),
                nn.Conv2d(hidden_size, 1, 1)
            )
        else:
            raise ValueError('unsupported mode')
        self.mode = mode

    def forward(self, code, seq):
        if self.mode == 'bilinear':
            u = torch.einsum('bg,gz,dzt->bdt', code, self.bilinear_weight, seq)
        elif self.mode == 'dot':
            u = torch.einsum('bg,dgt->bdt', code, seq)
        else:
            batch_size, _, seq_len = seq.size()
            code_expanded = code.unsqueeze(1).unsqueeze(3).expand(-1, batch_size, -1, seq_len)
            seq_expanded = seq.unsqueeze(0).expand(batch_size, -1, -1, -1)
            u = self.network(torch.cat([code_expanded, seq_expanded], dim=2).permute(0, 2, 1, 3)).squeeze()
        positive_mask = torch.eye(u.size(0)).to(u)
        return calc_one_way_loss(u, positive_mask, 2, self.measure), calc_accuracy(u)


class KPredLoss(nn.Module):
    def __init__(self, c_size, z_size, k=4, measure='JSD', mode='bilinear',
                 auto_is_bidirectional=True, look_both=False):
        super().__init__()
        assert k > 0
        self.k = k
        self.split_c = auto_is_bidirectional and not look_both
        self.bidirectional = auto_is_bidirectional
        measure = measure.upper()
        mode = mode.lower()
        self.measure = measure
        self.mode = mode

        c_size = (c_size // 2) if self.split_c else c_size
        if self.mode == 'bilinear':
            self.forward_weights = []
            for i in range(k):
                self.forward_weights.append(nn.Parameter(torch.randn(c_size, z_size)))
                self.register_parameter('forward_{}'.format(i), self.forward_weights[-1])
            if auto_is_bidirectional:
                self.backward_weights = []
                for i in range(k):
                    self.backward_weights.append(nn.Parameter(torch.randn(c_size, z_size)))
                    self.register_parameter('backward_{}'.format(i), self.backward_weights[-1])
        else:
            raise ValueError('only bilinear is supported')

    def calc_one_way_loss_t(self, u, t, positive_mask, mean_dim):
        positive_mask[:, t] = 1.0
        return calc_one_way_loss(u, positive_mask, mean_dim, self.measure)

    def calc_one_way_loss_b(self, u, b, positive_mask, mean_dim):
        positive_mask[b, :] = 1.0
        return calc_one_way_loss(u, positive_mask, mean_dim, self.measure)

    def calc_both_way_loss_t(self, c_flat, z_flat, weight):
        t = z_flat.size(2)
        t = np.random.randint(t)
        if self.mode == 'bilinear':
            u = torch.einsum('bct,cz,dz->btd', c_flat, weight, z_flat[..., t])
        else:
            u = torch.einsum('bzt,dz->btd', c_flat, z_flat[..., t])
        acc1 = (u.argmax(dim=1) == t).sum().item() / (u.size(0) * u.size(1))
        first_loss = self.calc_one_way_loss_t(u, t, torch.zeros(u.size(0), u.size(1)).to(u), 2)
        t = c_flat.size(2)
        t = np.random.randint(t)
        if self.mode == 'bilinear':
            u = torch.einsum('bc,cz,dzt->bdt', c_flat[..., t], weight, z_flat)
        else:
            u = torch.einsum('bz,dzt->bdt', c_flat[..., t], z_flat)
        acc2 = (u.argmax(dim=2) == t).sum().item() / (u.size(1) * u.size(2))
        return first_loss + self.calc_one_way_loss_t(u, t, torch.zeros(u.size(1), u.size(2)).to(u), 0), (acc1, acc2)

    def calc_both_way_loss_b(self, c_flat, z_flat, weight):
        b = z_flat.size(0)
        b = np.random.randint(b)
        if self.mode == 'bilinear':
            u = torch.einsum('bct,cz,zi->bti', c_flat, weight, z_flat[b])
        else:
            u = torch.einsum('bzt,zi->bti', c_flat, z_flat[b])
        acc1 = (u.argmax(dim=0) == b).sum().item() / (u.size(0) * u.size(1))
        first_loss = self.calc_one_way_loss_b(u, b, torch.zeros(u.size(0), u.size(1)).to(u), 2)
        b = c_flat.size(0)
        b = np.random.randint(b)
        if self.mode == 'bilinear':
            u = torch.einsum('ct,cz,dzi->tdi', c_flat[b], weight, z_flat)
        else:
            u = torch.einsum('zt,dzi->tdi', c_flat[b], z_flat)
        acc2 = (u.argmax(dim=1) == b).sum().item() / (u.size(1) * u.size(2))
        return first_loss + self.calc_one_way_loss_b(u, b, torch.zeros(u.size(1), u.size(2)).to(u), 0), (acc1, acc2)

    def calc_four_way_loss(self, c_flat, z_flat, weight):
        t_loss, t_acc = self.calc_both_way_loss_t(c_flat, z_flat, weight)
        b_loss, b_acc = self.calc_both_way_loss_b(c_flat, z_flat, weight)
        return t_loss + b_loss, t_acc, b_acc

    def forward(self, c, z):
        B, c_size, T = c.size()
        if self.split_c:
            c_size = c_size // 2
        total_loss = 0.0
        accs = {}
        for i in range(self.k):
            c_flat = c[:, :c_size, :-(i + 1)]
            z_flat = z[..., (i + 1):]
            f_loss, t_acc, b_acc = self.calc_four_way_loss(c_flat, z_flat, self.forward_weights[i])
            accs['f_{}_t'.format(i + 1)] = t_acc
            accs['f_{}_b'.format(i + 1)] = b_acc
            total_loss = total_loss + f_loss
        if self.bidirectional:
            for i in range(self.k):
                c_flat = c[:, -c_size:, (i + 1):]
                z_flat = z[..., :-(i + 1)]
                f_loss, t_acc, b_acc = self.calc_four_way_loss(c_flat, z_flat, self.backward_weights[i])
                accs['b_{}_t'.format(i + 1)] = t_acc
                accs['b_{}_b'.format(i + 1)] = b_acc
                total_loss = total_loss + f_loss
        return total_loss, accs
