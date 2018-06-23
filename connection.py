import torch
import torch.nn as nn
from utils import cudize, simple_argparser, enable_benchmark
import os
from tqdm import trange
from mmd import mix_rbf_mmd2
import torch.nn.functional as F


class ConnectionNet(nn.Module):
    # NOTE isn't it better to use Discriminator-like architecture here?
    # NOTE connection network can be trained with the GAN, real_loss += mmd only, fake_loss += mmd+mse
    def __init__(self, input_channels, input_len, output_size):
        super(ConnectionNet, self).__init__()
        self.net = [nn.Conv1d(input_channels, input_channels, kernel_size=5, padding=2)]
        self.net.append(nn.ReLU())
        self.net.append(nn.Conv1d(input_channels, input_channels * 2, kernel_size=3, padding=1, stride=2))
        self.net.append(nn.ReLU())
        self.net.append(nn.Conv1d(input_channels * 2, input_channels, kernel_size=3, padding=1, stride=2))
        self.net.append(nn.ReLU())
        self.net.append(nn.Conv1d(input_channels, input_channels * 2, kernel_size=3, padding=1))
        self.net.append(nn.ReLU())
        self.net.append(nn.Conv1d(input_channels * 2, input_channels, kernel_size=3, padding=1, stride=2))
        self.net.append(nn.ReLU())
        self.net = nn.Sequential(*self.net)
        self.linear = nn.Linear(input_len // 8 * input_channels, output_size)

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.linear(x)


if __name__ == '__main__':
    default_params = dict(
        snapshot_epoch=1000,
        minis=20000,
        batch_size=32,
        lr=0.001,
        checkpoints_path='results/exp_01',
        l_mean=0.01,
        l_std=0.001,
        l_mmd=1.0
    )
    params = simple_argparser(default_params)

    # sigma for MMD
    base = 1.0
    sigma_list = [1, 2, 4, 8, 16]
    sigma_list = [sigma / base for sigma in sigma_list]
    pattern = 'network-snapshot-{}-{}.dat'
    G = torch.load(os.path.join(params['checkpoints_path'], pattern.format('generator', params['snapshot_epoch'])),
                   map_location=lambda storage, location: storage)
    D = torch.load(os.path.join(params['checkpoints_path'], pattern.format('discriminator', params['snapshot_epoch'])),
                   map_location=lambda storage, location: storage)
    latent_size = G.latent_size
    sample = D(G(torch.randn(1, latent_size)), intermediate=True)
    num_channels = sample.size(1)
    seq_len = sample.size(2)
    D = cudize(D)
    G = cudize(G)
    D.eval()
    G.eval()
    loss_function = nn.MSELoss()
    C = cudize(ConnectionNet(num_channels, seq_len, latent_size))
    optimizer = torch.optim.Adam(C.parameters(), lr=params['lr'])
    enable_benchmark()
    for i in trange(params['minis']):
        z = cudize(torch.randn(params['batch_size'], latent_size))
        d = D(G(z), intermediate=True).detach()
        optimizer.zero_grad()
        p = C(d)
        loss = loss_function(p, z)
        loss = loss + params['l_mean'] * (p.mean(dim=-1) ** 2).mean() + params['l_std'] * ((p.std(dim=-1) - 1.0) ** 2).mean()
        loss = loss + params['l_mmd'] * torch.sqrt(F.relu(mix_rbf_mmd2(z, p, sigma_list)))
        loss.backward()
        optimizer.step()
        if i % 200 == 0:
            print(loss.item())
    torch.save(C, os.path.join(params['checkpoints_path'], 'connection_network.pth'))
