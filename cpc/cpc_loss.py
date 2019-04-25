import math
import torch
import numpy as np
import torch.nn as nn
from sklearn import metrics
import torch.nn.functional as F
from sklearn.utils.linear_assignment_ import linear_assignment


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
    def __init__(self, code_size, global_size, hidden_size, mode='bilinear', measure='FULL_CE'):
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
        if self.measure == 'FULL_CE':
            targets = torch.arange(u.size(0)).to(u.device)
            acc = (u.argmax(1) == targets).sum().item() / u.size(0)
            return F.cross_entropy(u, targets), acc
        positive_mask = torch.eye(u.size(0)).to(u)
        return calc_one_way_loss(u, positive_mask, None, self.measure), calc_accuracy(u)


class SeqOneMI(nn.Module):
    def __init__(self, code_size, seq_size, hidden_size, mode='bilinear', measure='FULL_CE'):
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
        if self.measure == 'FULL_CE':
            targets = torch.arange(u.size(0)).to(u.device).unsqueeze(1).expand(-1, u.size(2))  # BT
            acc = (u.argmax(1) == targets).sum().item() / (u.size(0) * u.size(2))
            return F.cross_entropy(u, targets), acc
        positive_mask = torch.eye(u.size(0)).to(u)
        return calc_one_way_loss(u, positive_mask, 2, self.measure), calc_accuracy(u)


class KPredLoss(nn.Module):
    def __init__(self, c_size, z_size, k=4, measure='FULL_CE', auto_is_bidirectional=True, look_both=False):
        super().__init__()
        if k <= 0:
            k = 0
        self.k = k
        self.split_c = auto_is_bidirectional and not look_both
        self.bidirectional = auto_is_bidirectional
        measure = measure.upper()
        self.measure = measure
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

    def calc_one_way_loss_t(self, u, t, positive_mask, mean_dim):
        positive_mask[:, t] = 1.0
        return calc_one_way_loss(u, positive_mask, mean_dim, self.measure)

    def calc_one_way_loss_b(self, u, b, positive_mask, mean_dim):
        positive_mask[b, :] = 1.0
        return calc_one_way_loss(u, positive_mask, mean_dim, self.measure)

    @staticmethod
    def calc_cross_entropy_loss(u, softmax_dim, target_id):
        if softmax_dim == 0:
            _u = u.permute(1, 0, 2)
        elif softmax_dim == 2:
            _u = u.permute(0, 2, 1)
        else:
            _u = u
        return F.cross_entropy(_u, (torch.ones(_u.size(0), _u.size(2)) * target_id).long().to(u.device))

    def calc_both_way_loss_t(self, c_flat, z_flat, weight):
        t = z_flat.size(2)
        t = np.random.randint(t)
        u = torch.einsum('bct,cz,dz->btd', c_flat, weight, z_flat[..., t])
        acc1 = (u.argmax(dim=1) == t).sum().item() / (u.size(0) * u.size(1))
        if self.measure == 'SAMPLED_CE':
            first_loss = self.calc_cross_entropy_loss(u, 1, t)
        else:
            first_loss = self.calc_one_way_loss_t(u, t, torch.zeros(u.size(0), u.size(1)).to(u), 2)
        t = c_flat.size(2)
        t = np.random.randint(t)
        u = torch.einsum('bc,cz,dzt->bdt', c_flat[..., t], weight, z_flat)
        acc2 = (u.argmax(dim=2) == t).sum().item() / (u.size(1) * u.size(2))
        if self.measure == 'SAMPLED_CE':
            second_loss = self.calc_cross_entropy_loss(u, 2, t)
        else:
            second_loss = self.calc_one_way_loss_t(u, t, torch.zeros(u.size(1), u.size(2)).to(u), 0)
        return first_loss + second_loss, (acc1 + acc2) / 2

    def calc_both_way_loss_b(self, c_flat, z_flat, weight):
        b = z_flat.size(0)
        b = np.random.randint(b)
        u = torch.einsum('bct,cz,zi->bti', c_flat, weight, z_flat[b])
        acc1 = (u.argmax(dim=0) == b).sum().item() / (u.size(0) * u.size(1))
        if self.measure == 'SAMPLED_CE':
            first_loss = self.calc_cross_entropy_loss(u, 0, b)
        else:
            first_loss = self.calc_one_way_loss_b(u, b, torch.zeros(u.size(0), u.size(1)).to(u), 2)
        b = c_flat.size(0)
        b = np.random.randint(b)
        u = torch.einsum('ct,cz,dzi->tdi', c_flat[b], weight, z_flat)
        acc2 = (u.argmax(dim=1) == b).sum().item() / (u.size(1) * u.size(2))
        if self.measure == 'SAMPLED_CE':
            second_loss = self.calc_cross_entropy_loss(u, 1, b)
        else:
            second_loss = self.calc_one_way_loss_b(u, b, torch.zeros(u.size(1), u.size(2)).to(u), 0)
        return first_loss + second_loss, (acc1 + acc2) / 2

    def calc_four_way_loss(self, c_flat, z_flat, weight):
        t_loss, t_acc = self.calc_both_way_loss_t(c_flat, z_flat, weight)
        b_loss, b_acc = self.calc_both_way_loss_b(c_flat, z_flat, weight)
        return t_loss + b_loss, t_acc, b_acc

    @staticmethod
    def calc_full_cross_entropy_loss(c_flat, z_flat, weights):
        c_flat = c_flat.contiguous().view(-1, c_flat.size(1))
        z_flat = z_flat.contiguous().view(-1, z_flat.size(1))
        u = torch.einsum('bc,cz,dz->bd', c_flat, weights, z_flat)
        targets = torch.arange(u.size(0)).to(u.device)
        acc = (u.argmax(1) == targets).sum().item() / u.size(0)
        f_loss = F.cross_entropy(u, targets)
        return f_loss, acc, acc

    def forward(self, c, z):
        if self.k == 0:
            return torch.tensor(0.0).to(z), {}
        B, c_size, T = c.size()
        if self.split_c:
            c_size = c_size // 2
        total_loss = 0.0
        accs = {}
        for i in range(self.k):
            c_flat = c[:, :c_size, :-(i + 1)]
            z_flat = z[..., (i + 1):]
            if self.measure == 'FULL_CE':
                f_loss, t_acc, b_acc = self.calc_full_cross_entropy_loss(c_flat, z_flat, self.forward_weights[i])
            else:
                f_loss, t_acc, b_acc = self.calc_four_way_loss(c_flat, z_flat, self.forward_weights[i])
            accs['f_{}'.format(i)] = (t_acc + b_acc) / 2
            total_loss = total_loss + f_loss
            if self.bidirectional:
                c_flat = c[:, -c_size:, (i + 1):]
                z_flat = z[..., :-(i + 1)]
                if self.measure == 'FULL_CE':
                    f_loss, t_acc, b_acc = self.calc_full_cross_entropy_loss(c_flat, z_flat, self.forward_weights[i])
                else:
                    f_loss, t_acc, b_acc = self.calc_four_way_loss(c_flat, z_flat, self.backward_weights[i])
                accs['b_{}'.format(i)] = (t_acc + b_acc) / 2
                total_loss = total_loss + f_loss
        return total_loss, accs


def _compute_joint(x_out, x_tf_out):
    # produces variable that requires grad (since args require grad)
    bn, k = x_out.size()
    assert (x_tf_out.size(0) == bn and x_tf_out.size(1) == k)
    p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  # bn, k, k
    p_i_j = p_i_j.sum(dim=0)  # k, k
    p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
    p_i_j = p_i_j / p_i_j.sum()  # normalise
    return p_i_j


def IIC(x_out, x_tf_out, eps=0.0001):  # x_out is n*C(softmaxed) and zt is it's pair
    _, k = x_out.size()
    p_i_j = _compute_joint(x_out, x_tf_out)
    assert (p_i_j.size() == (k, k))
    p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
    p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)  # but should be same, symmetric
    p_i_j[(p_i_j < eps).data] = eps
    p_j[(p_j < eps).data] = eps
    p_i[(p_i < eps).data] = eps
    loss = - p_i_j * (torch.log(p_i_j) - torch.log(p_j) - torch.log(p_i))
    loss = loss.sum()
    return loss


def myIIC(x_out, x_tf_out, eps=0.0001):  # x_out is n*C(softmaxed) and zt is it's pair
    _, k = x_out.size()
    p_i_j = _compute_joint(x_out, x_tf_out)
    assert (p_i_j.size() == (k, k))
    p_i_j[(p_i_j < eps).data] = eps
    return F.kl_div(p_i_j, target=torch.eye(k).to(x_out) / k, reduction='mean')


def _original_match(flat_preds, flat_targets, preds_k, targets_k):
    # map each output channel to the best matching ground truth (many to one)
    out_to_gts = {}
    out_to_gts_scores = {}
    for out_c in range(preds_k):
        for gt_c in range(targets_k):
            # the amount of out_c at all the gt_c samples
            tp_score = int(((flat_preds == out_c) * (flat_targets == gt_c)).sum())
            if (out_c not in out_to_gts) or (tp_score > out_to_gts_scores[out_c]):
                out_to_gts[out_c] = gt_c
                out_to_gts_scores[out_c] = tp_score
    return list(out_to_gts.items())


def _hungarian_match(flat_preds, flat_targets, num_k):
    num_samples = flat_targets.shape[0]
    num_correct = np.zeros((num_k, num_k))
    for c1 in range(num_k):
        for c2 in range(num_k):
            # elementwise, so each sample contributes once
            votes = int(((flat_preds == c1) * (flat_targets == c2)).sum())
            num_correct[c1, c2] = votes
    # num_correct is small
    return list(linear_assignment(num_samples - num_correct))


def _acc(preds, targets):
    return (preds == targets).mean()


def _nmi(preds, targets):
    return metrics.normalized_mutual_info_score(targets, preds)


def _ari(preds, targets):
    return metrics.adjusted_rand_score(targets, preds)


def calc_iic_stats(train_preds, val_preds, train_targets, val_targets, k):
    match_hun = _hungarian_match(train_preds, train_targets, k)
    match_ori = _original_match(train_preds, train_targets, preds_k=k, targets_k=k)
    acc = {}
    for i, match in enumerate([match_hun, match_ori, match_hun, match_ori]):
        preds = train_preds if i < 2 else val_preds
        reordered_preds = np.zeros_like(preds)
        for pred_i, target_i in match:
            reordered_preds[preds == pred_i] = target_i
        mode = 'train' if i < 2 else 'validation'
        targets = train_targets if i < 2 else val_targets
        if i % 2 == 0:
            acc[mode + '_hungarian'] = _acc(reordered_preds, targets)
        elif i % 2 == 1:
            acc[mode + '_original'] = _acc(reordered_preds, targets)
    return acc
