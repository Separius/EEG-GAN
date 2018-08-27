import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import cudize, enable_benchmark, load_model, save_pkl, simple_argparser


def run_static(gen, disc, pop_size=128, stage=3, ratio=0.95, y=None):
    latent_size = gen.latent_size
    z1 = cudize(torch.randn(1, latent_size))
    z2 = z1 * ratio + cudize(torch.randn(pop_size, latent_size)) * (1.0 - ratio)
    fake = gen.consistent_forward(z1, z2, stage=stage, y=y)
    scores = disc.consistent_forward(fake, y=y)
    return fake[scores.argmax().item(), ...], scores, z1, z2


def calc_channels(start, end, num_layers):
    step = pow(end / start, 1 / num_layers)
    res = []
    for i in range(num_layers + 1):
        s = round(start * step ** i)
        res.append(s)
    return res


class CompatNet(nn.Module):
    def __init__(self, latent_size, num_layers):
        super(CompatNet, self).__init__()
        net = []
        num_channels = calc_channels(latent_size * 2, 1, num_layers)
        for i in range(num_layers):
            net.append(nn.Linear(num_channels[i], num_channels[i + 1]))
            if i != num_layers - 1:
                net.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*net)

    def forward(self, z1, z2):  # z1 = N*latent, z2=N*2*latent
        z = torch.cat((z1, z2[:, 0, :]), dim=1)
        high = self.net(z).squeeze()
        z = torch.cat((z1, z2[:, 1, :]), dim=1)
        low = self.net(z).squeeze()
        return high, low


params = dict(
    checkpoints_path='results/001-test',
    pattern='network-snapshot-{}-{}.dat',
    snapshot_epoch='000040',
    dumb=True,
    pop_size=128,
    g_stage=3,
    static_ratio=0.95,
    num_classes=0,
    static_output_location=None,
    smart_db_size=1024 * 32
)
params = simple_argparser(params)

G = load_model(
    os.path.join(params['checkpoints_path'], params['pattern'].format('generator', params['snapshot_epoch'])))
D = load_model(
    os.path.join(params['checkpoints_path'], params['pattern'].format('discriminator', params['snapshot_epoch'])))

enable_benchmark()
G = cudize(G)
D = cudize(D)
G.eval()
D.eval()

if params['dumb']:
    generated, scores, z1, z2 = run_static(G, D, pop_size=params['pop_size'], stage=params['g_stage'],
                                           ratio=params['static_ratio'])
    print('min:', scores.min().item(), 'max:', scores.max().item(), 'mean:', scores.mean().item())
    generated = generated.unsqueeze(0).data.cpu().numpy()
    if params['static_output_location'] is None:
        loc = os.path.join(params['checkpoints_path'], 'static_generated.pkl')
    else:
        loc = params['static_output_location']
    save_pkl(loc, generated)
else:
    db_z1 = []
    db_z2 = []
    for i in range(params['smart_db_size']):
        generated, scores, z1, z2 = run_static(G, D, pop_size=params['pop_size'], stage=params['g_stage'],
                                               ratio=params['static_ratio'])
        db_z1.append(z1)  # ((1, latent_size), (2, latent_size))
        db_z2.append(z2[torch.topk(scores, 2)[1]])
    bs = params['smart_batch_size']
    target = cudize(torch.ones(bs))
    network = cudize(CompatNet(G.latent_size, params['smart_num_layers']))
    optimizer = torch.optim.Adam(network.params(), lr=params['smart_lr'])
    for i in range(len(db_z1) // bs):
        mini_batch = torch.cat(db_z1[i * bs:(i + 1) * bs], dim=0), torch.cat(
            [x.unsqueeze(0) for x in db_z2[i * bs:(i + 1) * bs]], dim=0)
        optimizer.zero_grad()
        high, low = network(*mini_batch)
        loss = F.margin_ranking_loss(high, low, target)
        loss.backward()
        optimizer.step()
    torch.save(network, os.path.join(params['checkpoints_path'], 'compatibility_network.pth'))  # TODO use it
