import torch
import os
import numpy as np
from utils import cudize, load_pkl, simple_argparser, enable_benchmark, load_model, save_pkl, num_params, \
    trainable_params
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataset import EEGDataset
from torch.utils.data.sampler import RandomSampler
import torch.nn as nn
from tqdm import tqdm
from connection import ConnectionNet  # NOTE this is necessary for the python


class CompatibleDataset(Dataset):
    def __init__(self, eeg_dataset, generator, discriminator, is_fisrt=True, negative_samples=16,
                 connection_network=None):
        super(CompatibleDataset, self).__init__()
        self.eeg_dataset = eeg_dataset
        self.generator = generator
        self.discriminator = discriminator
        self.negative_samples = negative_samples
        self.connection_network = connection_network
        self.seq_len = eeg_dataset.seq_len // 2
        self.is_first = is_fisrt

    def negative_sample_generator(self, ans):
        res = [ans.unsqueeze(0)]
        z_dim = self.generator.latent_size
        for i in range(self.negative_samples - 1):
            if self.connection_network is not None:
                if np.random.rand() < 0.2:
                    if params['cudify_dataset']:
                        res.append(cudize(torch.randn(1, z_dim)))
                    else:
                        res.append(torch.randn(1, z_dim))
                    continue
            if np.random.rand() < 0.5:
                data = self.eeg_dataset[int(np.random.rand() * len(self))][:, :self.seq_len]
            else:
                z = torch.randn(1, z_dim)
                if params['cudify_dataset']:
                    z = cudize(z)
                data = self.generator(z).detach().squeeze()
            res.append(self.feature_extract(data).unsqueeze(0))
        return torch.cat(res, dim=0)

    def feature_extract(self, data):
        intermediate = self.discriminator(data.unsqueeze(0), intermediate=True)
        if self.connection_network is not None:
            return self.connection_network(intermediate).view(-1).detach()
        return intermediate.view(-1).detach()

    def __getitem__(self, index):
        real_data = self.eeg_dataset[index]  # tensor: ch, seq_len
        p1 = self.feature_extract(real_data[:, :self.seq_len])
        p2 = self.feature_extract(real_data[:, self.seq_len:])
        if self.is_first:
            p1 = p1
            p2 = self.negative_sample_generator(p2)
        else:
            p1 = self.negative_sample_generator(p1)
            p2 = p2
        return (p1, p2)

    def __len__(self):
        return len(self.eeg_dataset)


def run_static(gen, disc, pop_size=64, stage=2, y=None, ratio=0.95):
    latent_size = gen.latent_size
    z1 = cudize(torch.randn(1, latent_size))
    z2 = z1 * ratio + cudize(torch.randn(pop_size, latent_size)) * (1.0 - ratio)
    fake = gen.consistent_forward(z1, z2, stage=stage, y=y)
    scores = disc.consistent_forward(fake, y=y)
    return fake[scores.argmax().item(), ...], scores


def score_real_data(disc, real, y=None):
    return disc.consistent_forward(real, y=y)


def margin_ranking_rank(right, wrong, margin=0):
    return F.margin_ranking_loss(right, wrong, cudize(torch.ones(right.size(0)).long()), margin)


def pick_loss(pred):
    return F.cross_entropy(pred, cudize(torch.zeros(pred.size(0)).long()))


def calc_channels(start, end, num_layers):
    step = pow(end / start, 1 / num_layers)
    res = []
    for i in range(num_layers + 1):
        s = round(start * step ** i)
        res.append(s)
    return res


class CompatNet(nn.Module):
    def __init__(self, input_size, num_layers=3, dim_size=32):
        super(CompatNet, self).__init__()
        num_channels = calc_channels(input_size, dim_size, num_layers)
        self.fc1 = nn.Sequential(*[nn.Linear(num_channels[i], num_channels[i + 1]) for i in range(num_layers)])
        self.fc2 = nn.Sequential(*[nn.Linear(num_channels[i], num_channels[i + 1]) for i in range(num_layers)])

    def forward(self, x1, x2, is_first=True):
        if is_first:
            fc1 = self.fc1
            fc2 = self.fc2
        else:
            fc1 = self.fc2
            fc2 = self.fc1
            x2, x1 = x1, x2
        # NOTE that is assumes: x1 is NxIn ; x2 is NxKxIn
        k = x2.size(1)
        x2 = x2.view(-1, x2.size(2))
        x1 = fc1(x1)
        x2 = fc2(x2)
        x2 = x2.view(x1.size(0), k, -1)
        x1 = x1.view(x1.size(0), 1, -1)
        return (x1 * x2).sum(-1)


class InfiniteRandomSampler(RandomSampler):

    def __iter__(self):
        while True:
            it = super().__iter__()
            for x in it:
                yield x


if __name__ == '__main__':
    default_params = dict(
        checkpoints_path='results/001-test',
        snapshot_epoch='000040',
        pattern='network-snapshot-{}-{}.dat',
        is_static=False,
        eeg_dataset_address='./data/test512.pkl',
        batch_size=16,
        negative_samples=4,
        is_connection=True,
        cudify_dataset=False,
        frequency=80,
        minis=2000,
        log_step=50
    )
    params = simple_argparser(default_params)

    G = load_model(
        os.path.join(params['checkpoints_path'], params['pattern'].format('generator', params['snapshot_epoch'])))
    D = load_model(
        os.path.join(params['checkpoints_path'], params['pattern'].format('discriminator', params['snapshot_epoch'])))
    if params['is_static']:
        G = cudize(G)
        D = cudize(D)
        G.eval()
        D.eval()
        generated, scores = run_static(G, D)
        print('min:', scores.min().item(), 'max:', scores.max().item(), 'mean:', scores.mean().item())
        generated = generated.unsqueeze(0).data.cpu().numpy()
        save_pkl(os.path.join(params['checkpoints_path'], 'static_compat_generated.pkl'), generated)
    else:
        if params['cudify_dataset']:
            G = cudize(G)
            D = cudize(D)
        G.eval()
        D.eval()
        eeg_dataset_base = load_pkl(params['eeg_dataset_address'])
        eeg_dataset_base.model_depth = eeg_dataset_base.max_dataset_depth - eeg_dataset_base.model_dataset_depth_offset
        eeg_dataset_base.alpha = 1
        progression_scale = eeg_dataset_base.progression_scale
        if isinstance(eeg_dataset_base.progression_scale, list):
            progression_scale = progression_scale + [2]
        if isinstance(eeg_dataset_base.progression_scale, tuple):
            progression_scale = list(progression_scale) + [2]
        eeg_dataset = EEGDataset(progression_scale, eeg_dataset_base.dir_path,
                                 num_files=len(eeg_dataset_base.all_files), seq_len=eeg_dataset_base.seq_len * 2,
                                 stride=eeg_dataset_base.stride, max_freq=eeg_dataset_base.max_freq,
                                 num_channels=eeg_dataset_base.num_channels, per_user=eeg_dataset_base.per_user,
                                 use_abs=eeg_dataset_base.use_abs, dataset_freq=eeg_dataset_base.dataset_freq,
                                 model_dataset_depth_offset=eeg_dataset_base.model_dataset_depth_offset)
        eeg_dataset.model_depth = eeg_dataset_base.model_depth + 1
        eeg_dataset.alpha = 1
        if params['is_connection']:
            C = load_model(os.path.join(params['checkpoints_path'], 'connection_network.pth'))
            if params['cudify_dataset']:
                C = cudize(C)
            C.eval()
        else:
            C = None
        cds1 = CompatibleDataset(eeg_dataset, G, D, is_fisrt=True, negative_samples=params['negative_samples'],
                                 connection_network=C)
        cds2 = CompatibleDataset(eeg_dataset, G, D, is_fisrt=False, negative_samples=params['negative_samples'],
                                 connection_network=C)
        dl1 = DataLoader(cds1, params['batch_size'], sampler=InfiniteRandomSampler(cds1), num_workers=2,
                         pin_memory=False, drop_last=True)
        dl2 = DataLoader(cds2, params['batch_size'], sampler=InfiniteRandomSampler(cds1), num_workers=2,
                         pin_memory=False, drop_last=True)
        dl1i = dl1.__iter__()
        sample = dl1i.next()
        dl2i = dl2.__iter__()
        net = cudize(CompatNet(sample[0].size(1)))
        optimizer = torch.optim.Adam(trainable_params(net), lr=0.001)
        enable_benchmark()
        mini_tqdm = tqdm(range(params['minis']))
        for i in range(params['minis']):
            is_first = np.random.rand() < 0.5
            x1, x2 = dl1i.next() if is_first else dl2i.next()
            if params['cudify_dataset']:
                o = net(x1, x2, is_first)
            else:
                o = net(cudize(x1), cudize(x2), is_first)
            l = pick_loss(o)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            if i % params['log_step'] == 0:
                mini_tqdm.set_description('loss: ' + str(l.item()))
            mini_tqdm.update()
        mini_tqdm.close()
        torch.save(net, os.path.join(params['checkpoints_path'], 'compatibility_network.pth'))
