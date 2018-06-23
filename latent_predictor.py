import torch
import torch.nn as nn
from utils import cudize, simple_argparser, enable_benchmark, load_model, trainable_params, num_params, save_pkl
import os
from tqdm import trange, tqdm


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
        snapshot_epoch='000040',
        minis=2000,
        batch_size=32,
        lr=0.001,
        checkpoints_path='results/001-test',
        pattern='network-snapshot-{}-{}.dat',
        log_step=50
    )
    params = simple_argparser(default_params)

    G = load_model(
        os.path.join(params['checkpoints_path'], params['pattern'].format('generator', params['snapshot_epoch'])))
    latent_size = G.latent_size
    print('latent size:', latent_size)
    sample = G(torch.randn(1, latent_size), intermediate=True)
    print('generator intermediate result:', sample.shape)
    num_channels = sample.size(1)
    seq_len = sample.size(2)
    G = cudize(G)
    G.eval()
    loss_function = nn.MSELoss()
    C = cudize(PredictorNet(num_channels, seq_len))
    print('number of parameters:', num_params(C))
    optimizer = torch.optim.Adam(trainable_params(C), lr=params['lr'])
    enable_benchmark()
    mini_tqdm = tqdm(range(params['minis']))
    for i in range(params['minis']):
        z = cudize(torch.randn(params['batch_size'], latent_size))
        d = G(z, intermediate=True).detach()
        optimizer.zero_grad()
        loss = loss_function(C(d[:, :, :-1].contiguous()), d[:, :, -1])
        loss.backward()
        optimizer.step()
        if i % params['log_step'] == 0:
            mini_tqdm.set_description('loss: '+str(loss.item()))
        mini_tqdm.update()
    mini_tqdm.close()
    torch.save(C, os.path.join(params['checkpoints_path'], 'predictor_network.pth'))
    z = cudize(torch.randn(params['batch_size'] * 4, latent_size))
    C.eval()
    h = G(z, intermediate=True).detach()
    for _ in trange(seq_len * 4):
        n = C(h[:, :, -(seq_len - 1):].contiguous())
        h = torch.cat((h, n.unsqueeze(2)), dim=2)
    generated = G.after_first(h).data.cpu().numpy()
    save_pkl(os.path.join(params['checkpoints_path'], 'predictor_generated.pkl'), generated)
