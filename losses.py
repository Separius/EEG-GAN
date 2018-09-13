import torch
import random
from utils import cudize
import torch.nn.functional as F
from torch.autograd import Variable, grad

mixing_factors = None
grad_outputs = None
one = None
zero = None


def get_mixing_factor(batch_size):
    global mixing_factors
    if mixing_factors is None or batch_size != mixing_factors.size(0):
        mixing_factors = cudize(torch.FloatTensor(batch_size, 1, 1))
    mixing_factors.uniform_()
    return mixing_factors


def get_one(batch_size):
    global one
    if one is None or batch_size != one.size(0):
        one = Variable(cudize(torch.ones(batch_size)))
    return one


def get_zero(batch_size):
    global zero
    if zero is None or batch_size != zero.size(0):
        zero = Variable(cudize(torch.zeros(batch_size)))
    return zero


def calc_grad(x_hat, pred_hat):
    global grad_outputs
    if grad_outputs is None or pred_hat.size(0) != grad_outputs.size(0):
        grad_outputs = cudize(torch.ones(pred_hat.size()))
    return grad(outputs=pred_hat, inputs=x_hat, grad_outputs=grad_outputs, create_graph=True, retain_graph=True,
                only_inputs=True)[0]


def generator_loss(dis: torch.nn.Module, gen: torch.nn.Module, real: torch.tensor,
                   fake: torch.tensor, loss_type: str, random_multiply: bool):
    gen.zero_grad()
    z = Variable(fake)
    g_ = gen(z)
    d_fake, _ = dis(g_)
    scale = random.random() if random_multiply else 1.0
    if loss_type == 'hinge' or loss_type.startswith('wgan'):
        g_loss = -d_fake.mean()
    else:
        with torch.no_grad():
            x_real = Variable(real)
            d_real, _ = dis(x_real)
        if loss_type == 'rsgan':
            g_loss = F.binary_cross_entropy_with_logits(d_fake - d_real, get_one(d_fake.size(0)))
        elif loss_type == 'rasgan':
            batch_size = d_fake.size(0)
            g_loss = (F.binary_cross_entropy_with_logits(d_fake - d_real.mean(),
                                                         get_one(batch_size)) + F.binary_cross_entropy_with_logits(
                d_real - d_fake.mean(), get_zero(batch_size))) / 2.0
        elif loss_type == 'rahinge':
            g_loss = (torch.mean(F.relu(1.0 + (d_real - torch.mean(d_fake)))) + torch.mean(
                F.relu(1.0 - (d_fake - torch.mean(d_real))))) / 2
        else:
            raise ValueError('Invalid loss type')
    return g_loss * scale


def discriminator_loss(dis: torch.nn.Module, gen: torch.nn.Module, real: torch.tensor, fake: torch.tensor,
                       loss_type: str, iwass_epsilon: float, grad_lambda: float, iwass_target: float):
    dis.zero_grad()
    gen.zero_grad()
    x_real = Variable(real)
    d_real, d_last_real = dis(x_real)
    with torch.no_grad():
        z = Variable(fake)
    g_ = Variable(gen(z).data)
    d_fake, _ = dis(g_)
    batch_size = d_real.size(0)
    gp_gain = 1.0 if grad_lambda != 0 else 0
    if loss_type == 'hinge':
        d_loss = F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()
    elif loss_type == 'rsgan':
        d_loss = F.binary_cross_entropy_with_logits(d_real - d_fake, get_one(batch_size))
    elif loss_type == 'rasgan':
        d_loss = (F.binary_cross_entropy(d_real - d_fake.mean(), get_one(batch_size)) + F.binary_cross_entropy(
            d_fake - d_real.mean(), get_zero(batch_size))) / 2.0
    elif loss_type == 'rahinge':
        d_loss = (torch.mean(F.relu(1.0 - (d_real - torch.mean(d_fake)))) + torch.mean(
            F.relu(1.0 + (d_fake - torch.mean(d_real))))) / 2
    elif loss_type.startswith('wgan'):  # wgan and wgan_theirs
        d_fake_mean = d_fake.mean()
        d_real_mean = d_real.mean()
        if loss_type == 'wgan_theirs':
            d_loss = d_fake_mean - d_real_mean + (d_fake_mean + d_real_mean) ** 2 * iwass_epsilon
            gp_gain = F.relu(d_real_mean - d_fake_mean)
        elif loss_type == 'wgan_gp':
            d_loss = d_fake_mean - d_real_mean + (d_real ** 2).mean() * iwass_epsilon
            gp_gain = 1
        else:
            raise ValueError('Invalid loss type')
    else:
        raise ValueError('Invalid loss type')
    if gp_gain != 0:
        alpha = get_mixing_factor(x_real.size(0))
        min_size = min(g_.size(2), x_real.size(2))
        x_hat = Variable(alpha * x_real[:, :, :min_size].data + (1.0 - alpha) * g_[:, :, :min_size].data,
                         requires_grad=True)
        pred_hat, _ = dis(x_hat)
        g = calc_grad(x_hat, pred_hat).view(batch_size, -1)
        gp = g.norm(p=2, dim=1) - iwass_target
        if loss_type == 'wgan_theirs':
            gp = F.relu(gp)
        d_loss = d_loss + gp_gain * (gp ** 2).mean() * grad_lambda / (iwass_target ** 2)
    return d_loss
