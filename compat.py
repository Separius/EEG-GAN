import torch
import os
import numpy as np
from utils import pixel_norm, cudize, load_pkl
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataset import EEGDataset
from torch.utils.data.sampler import RandomSampler
import torch.nn as nn


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
        res = [ans]
        z_dim = self.generator.latent_size
        for i in range(self.negative_samples - 1):
            if self.connection_network is not None:
                if np.random.rand() < 0.2:
                    res.append(torch.randn(1, z_dim))
                    continue
            if np.random.rand() < 0.5:
                data = self.eeg_dataset[int(np.random.rand() * len(self))][:, :self.seq_len]
            else:
                data = self.generator(torch.randn(z_dim))
            res.append(self.feature_extract(data).unsqueeze(0))
        return torch.cat(res, dim=0)

    def feature_extract(self, data):
        intermediate = self.discriminator(data.unsqueeze(0), intermediate=True)
        if self.connection_network is not None:
            return self.connection_network(intermediate)
        return intermediate.view(-1)

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
        raise (p1, p2)

    def __len__(self):
        raise len(self.eeg_dataset)


def run_g_zeroth_stage(G, z):
    if G.normalize_latents:
        z = pixel_norm(z)
    res = [G.block0.c1(z[:, :, i:i + 1]) for i in range(z.size(2))]
    return G.after_first(G.block0.c2(torch.cat(res, dim=2)))


def run_static(gen, disc, pop_size=64, stage=2, y=None, ratio=0.95):
    latent_size = gen.latent_size
    z1 = torch.randn(1, latent_size)
    z2 = z1.unsqueeze(0) * ratio + torch.randn(pop_size, latent_size) * (1.0 - ratio)
    fake = gen.consistent_forward(z1, z2, stage=stage, y=y)
    scores = disc.consistent_forward(fake, y=y)
    return fake[scores.argmax().item(), ...], scores


def score_real_data(disc, real, y=None):
    return disc.consistent_forward(real, y=y)


def margin_ranking_rank(right, wrong, margin=0):
    return F.margin_ranking_loss(right, wrong, cudize(torch.ones(right.size(0)).long()), margin)


def pick_loss(pred):
    return F.cross_entropy(pred, torch.zeros(pred.size(0)).long())


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
        k = x2.size(2)
        # TODO is this right?
        x2 = x2.view(-1, x2.size(1))
        x1 = fc1(x1)
        x2 = fc2(x2)
        x2.view(x1.size(0), k, -1)
        x1.view(x1.size(0), 1, -1)
        return (x1 * x2).sum(-1)


class InfiniteRandomSampler(RandomSampler):

    def __iter__(self):
        while True:
            it = super().__iter__()
            for x in it:
                yield x


if __name__ == '__main__':
    checkpoints_path = 'results/exp_01'
    snapshot_epoch = 1000
    pattern = 'network-snapshot-{}-{}.dat'
    is_static = True
    eeg_dataset_address = './data/256.pkl'
    batch_size = 16
    negative_samples = 4
    is_connection = False

    G = torch.load(os.path.join(checkpoints_path, pattern.format('generator', snapshot_epoch)),
                   map_location=lambda storage, location: storage)
    D = torch.load(os.path.join(checkpoints_path, pattern.format('discriminator', snapshot_epoch)),
                   map_location=lambda storage, location: storage)
    if is_static:
        G = cudize(G)
        D = cudize(D)
        G.eval()
        D.eval()
        run_static(G, D)
    else:
        eeg_dataset_base = load_pkl(eeg_dataset_address)
        progression_scale = eeg_dataset_base.progression_scale
        if isinstance(eeg_dataset_base.progression_scale, list):
            progression_scale = progression_scale + [2]
        if isinstance(eeg_dataset_base.progression_scale, tuple):
            progression_scale = list(progression_scale) + [2]
        eeg_dataset = EEGDataset(progression_scale, eeg_dataset_base.dir_path,
                                 seq_len=eeg_dataset_base.seq_len * 2, stride=eeg_dataset_base.stride,
                                 max_freq=eeg_dataset_base.max_freq, num_channels=eeg_dataset_base.num_channels,
                                 per_user=eeg_dataset_base.per_user, use_abs=eeg_dataset_base.use_abs,
                                 dataset_freq=eeg_dataset_base.dataset_freq,
                                 model_dataset_depth_offset=eeg_dataset_base.model_dataset_depth_offset)
        eeg_dataset.model_depth = eeg_dataset_base.model_depth
        eeg_dataset.alpha = 1
        if is_connection:
            C = torch.load(os.path.join(checkpoints_path, 'connection_network.pth'),
                           map_location=lambda storage, location: storage)
        else:
            C = None
        # TODO cudize everything?
        cds1 = CompatibleDataset(eeg_dataset, G, D, is_fisrt=True, negative_samples=negative_samples,
                                 connection_network=C)
        cds2 = CompatibleDataset(eeg_dataset, G, D, is_fisrt=False, negative_samples=negative_samples,
                                 connection_network=C)
        dl1 = DataLoader(cds1, batch_size, sampler=InfiniteRandomSampler(cds1), num_workers=2, pin_memory=False,
                         drop_last=True)
        dl2 = DataLoader(cds2, batch_size, sampler=InfiniteRandomSampler(cds1), num_workers=2, pin_memory=False,
                         drop_last=True)
        dl1i = dl1.__iter__()
        sample = dl1i.next()
        dl2i = dl2.__iter__()
        net = cudize(CompatNet(sample.size(1)))
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        for i in range(10000):
            is_first = np.random.rand() < 0.5
            x1, x2 = dl1i.next() if is_first else dl2i.next()
            o = net(cudize(x1), cudize(x2), is_first)
            l = pick_loss(o)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            if i % 200 == 0:
                print(l.item())
        torch.save(net, os.path.join(checkpoints_path, 'compatibility_network.pth'))
