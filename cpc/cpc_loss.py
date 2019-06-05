import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def calc_slice_accuracy(u, bs):
    return ((u.argmax(dim=0).cpu() == torch.arange(bs)).sum().item() / bs + (
            u.argmax(dim=1).cpu() == torch.arange(bs)).sum().item() / bs) / 2


def calc_accuracy(u):
    bs = u.size(0)
    nd = u.ndimension()
    if nd == 2:
        return calc_slice_accuracy(u, bs)
    elif nd == 3:
        t = u.size(2)
        res = [calc_slice_accuracy(u[..., i], bs) for i in range(t)]
        return sum([r for r in res]) / t


class OneOneMI(nn.Module):
    def __init__(self, code_size, global_size, mode='bilinear'):
        super().__init__()
        mode = mode.lower()
        if mode == 'bilinear':
            self.bilinear_weight = nn.Parameter(torch.randn(code_size, global_size))
        elif mode == 'dot':
            assert code_size == global_size
        elif mode == 'mlp' or mode == 'cnn':
            hidden_size = min(code_size, global_size) * 2
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
        targets = torch.arange(u.size(0)).to(u.device)
        acc = (u.argmax(1) == targets).sum().item() / u.size(0)
        return F.cross_entropy(u, targets), acc


class SeqOneMI(nn.Module):
    def __init__(self, code_size, seq_size, mode='bilinear'):
        super().__init__()
        mode = mode.lower()
        if mode == 'bilinear':
            self.bilinear_weight = nn.Parameter(torch.randn(code_size, seq_size))
        elif mode == 'dot':
            assert code_size == seq_size
        elif mode == 'mlp' or mode == 'cnn':
            hidden_size = min(code_size, seq_size) * 2
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
        targets = torch.arange(u.size(0)).to(u.device).unsqueeze(1).expand(-1, u.size(2))  # BT
        acc = (u.argmax(1) == targets).sum().item() / (u.size(0) * u.size(2))
        return F.cross_entropy(u, targets), acc


class KPredLoss(nn.Module):
    def __init__(self, c_size, z_size, k=4, full_cross_entropy=True, auto_is_bidirectional=True, look_both=False):
        super().__init__()
        k = max(k, 0)
        self.k = k
        self.split_c = auto_is_bidirectional and not look_both
        self.bidirectional = auto_is_bidirectional
        self.full_cross_entropy = full_cross_entropy
        c_size = (c_size // 2) if self.split_c else c_size
        self.forward_weights = []
        for i in range(k):
            self.forward_weights.append(nn.Parameter(torch.randn(c_size, z_size)))
            self.register_parameter('forward_{}'.format(i), self.forward_weights[-1])
        if auto_is_bidirectional:
            self.backward_weights = []
            for i in range(k):
                self.backward_weights.append(nn.Parameter(torch.randn(c_size, z_size)))
                self.register_parameter('backward_{}'.format(i), self.backward_weights[-1])

    @staticmethod
    def cross_entropy_loss(u, softmax_dim, target_id):
        if softmax_dim == 0:
            _u = u.permute(1, 0, 2)
        elif softmax_dim == 2:
            _u = u.permute(0, 2, 1)
        else:
            _u = u
        return F.cross_entropy(_u, (torch.ones(_u.size(0), _u.size(2)) * target_id).long().to(u.device))

    def calc_sampled_cross_entropy_loss_t(self, c_flat, z_flat, weight):
        t = z_flat.size(2)
        t = np.random.randint(t)
        u = torch.einsum('bct,cz,dz->btd', c_flat, weight, z_flat[..., t])
        acc1 = (u.argmax(dim=1) == t).sum().item() / (u.size(0) * u.size(1))
        first_loss = self.cross_entropy_loss(u, 1, t)
        t = c_flat.size(2)
        t = np.random.randint(t)
        u = torch.einsum('bc,cz,dzt->bdt', c_flat[..., t], weight, z_flat)
        acc2 = (u.argmax(dim=2) == t).sum().item() / (u.size(1) * u.size(2))
        second_loss = self.cross_entropy_loss(u, 2, t)
        return first_loss + second_loss, (acc1 + acc2) / 2

    def calc_sampled_cross_entropy_loss_b(self, c_flat, z_flat, weight):
        b = z_flat.size(0)
        b = np.random.randint(b)
        u = torch.einsum('bct,cz,zi->bti', c_flat, weight, z_flat[b])
        acc1 = (u.argmax(dim=0) == b).sum().item() / (u.size(0) * u.size(1))
        first_loss = self.cross_entropy_loss(u, 0, b)
        b = c_flat.size(0)
        b = np.random.randint(b)
        u = torch.einsum('ct,cz,dzi->tdi', c_flat[b], weight, z_flat)
        acc2 = (u.argmax(dim=1) == b).sum().item() / (u.size(1) * u.size(2))
        second_loss = self.cross_entropy_loss(u, 1, b)
        return first_loss + second_loss, (acc1 + acc2) / 2

    def sampled_cross_entropy_loss(self, c_flat, z_flat, weight):
        t_loss, t_acc = self.calc_sampled_cross_entropy_loss_t(c_flat, z_flat, weight)
        b_loss, b_acc = self.calc_sampled_cross_entropy_loss_b(c_flat, z_flat, weight)
        return t_loss + b_loss, (t_acc + b_acc) / 2

    @staticmethod
    def full_cross_entropy_loss(c_flat, z_flat, weights):
        c_flat = c_flat.contiguous().view(-1, c_flat.size(1))
        z_flat = z_flat.contiguous().view(-1, z_flat.size(1))
        u = torch.einsum('bc,cz,dz->bd', c_flat, weights, z_flat)
        targets = torch.arange(u.size(0)).to(u.device)
        acc = (u.argmax(1) == targets).sum().item() / u.size(0)
        f_loss = F.cross_entropy(u, targets)
        return f_loss, acc

    def forward(self, c, z):
        if self.k == 0:
            return torch.tensor(0.0).to(z), {}
        B, c_size, T = c.size()
        if self.split_c:
            c_size = c_size // 2
        total_loss = 0.0
        accs = {}
        loss_func = self.full_cross_entropy_loss if self.full_cross_entropy else self.sampled_cross_entropy_loss
        for i in range(self.k):
            c_flat = c[:, :c_size, :-(i + 1)]
            z_flat = z[..., (i + 1):]
            f_loss, acc = loss_func(c_flat, z_flat, self.forward_weights[i])
            accs['f_{}'.format(i)] = acc
            total_loss = total_loss + f_loss
            if self.bidirectional:
                c_flat = c[:, -c_size:, (i + 1):]
                z_flat = z[..., :-(i + 1)]
                b_loss, acc = loss_func(c_flat, z_flat, self.backward_weights[i])
                accs['b_{}'.format(i)] = acc
                total_loss = total_loss + b_loss
        return total_loss, accs


# TODO use for MINE evaluation metric
def mutual_information(joint, marginal, mine_net):
    mean_t = mine_net(joint).mean()
    mean_et = torch.exp(mine_net(marginal)).mean()
    mi_lb = mean_t - torch.log(mean_et)
    return mi_lb, mean_t, mean_et


def learn_mine(batch, mine_net, mine_net_optim, ma_et=1.0, ma_rate=0.01, biased=False):
    joint, marginal = batch  # batch is a tuple of (joint, marginal)
    mi_lb, mean_t, mean_et = mutual_information(joint, marginal, mine_net)
    ma_et = (1.0 - ma_rate) * ma_et + ma_rate * mean_et.item()
    if not biased:
        loss = -(mean_t - mean_et / ma_et)
    else:
        loss = - mi_lb
    mine_net_optim.zero_grad()
    loss.backward()
    mine_net_optim.step()
    return mi_lb, ma_et


def sample_batch(data, batch_size=128):
    joint_index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
    marginal_index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
    return data[joint_index], data[marginal_index]
