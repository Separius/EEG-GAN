import torch
import torch.nn as nn
from utils import cudize, simple_argparser, enable_benchmark
import os
from tqdm import trange
import pickle


class PredictorNet(nn.Module):
    def __init__(self, input_channels, input_len):
        super(PredictorNet, self).__init__()
        in_channels = input_channels * (input_len - 1)
        self.net = nn.Sequential(nn.Linear(in_channels, in_channels * 2), nn.ReLU(),
                                 nn.Linear(in_channels * 2, in_channels), nn.ReLU(),
                                 nn.Linear(in_channels, input_channels))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)


if __name__ == '__main__':
    default_params = dict(
        snapshot_epoch=1000,
        minis=20000,
        batch_size=32,
        lr=0.001,
        checkpoints_path='results/exp_01',
        pattern='network-snapshot-{}-{}.dat'
    )
    params = simple_argparser(default_params)

    G = torch.load(os.path.join(params['checkpoints_path'], params['pattern'].format('generator', params['snapshot_epoch'])),
                   map_location=lambda storage, location: storage)
    latent_size = G.latent_size
    sample = G(torch.randn(1, latent_size), intermediate=True)
    num_channels = sample.size(1)
    seq_len = sample.size(2)
    G = cudize(G)
    G.eval()
    loss_function = nn.MSELoss()
    C = cudize(PredictorNet(num_channels, seq_len))
    optimizer = torch.optim.Adam(C.parameters(), lr=params['lr'])
    enable_benchmark()
    for i in trange(params['minis']):
        z = cudize(torch.randn(params['batch_size'], latent_size))
        d = G(z, intermediate=True).detach()
        optimizer.zero_grad()
        loss = loss_function(C(d[:, :, :-1]), d[:, :, -1])
        loss.backward()
        optimizer.step()
        if i % 200 == 0:
            print(loss.item())
    torch.save(C, os.path.join(params['checkpoints_path'], 'predictor_network.pth'))
    z = cudize(torch.randn(params['batch_size'] * 4, latent_size))
    C.eval()
    h = G(z, intermediate=True).detach()
    for _ in range(seq_len * 4):
        n = C(h[:, :, -(seq_len - 1):])
        h = torch.cat((h, n.unsqueeze(2)), dim=2)
    generated = G.after_first(h).data.cpu().numpy()
    pickle.dump(generated, open(os.path.join(params['checkpoints_path'], 'predictor_generated.pkl'), 'rb'))
