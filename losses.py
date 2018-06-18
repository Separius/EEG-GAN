import torch
from torch.autograd import Variable, grad
from utils import cudize
import torch.nn.functional as F

mixing_factors = None
grad_outputs = None


def get_mixing_factor(batch_size):
    global mixing_factors
    if mixing_factors is None or batch_size != mixing_factors.size(0):
        mixing_factors = cudize(torch.FloatTensor(batch_size, 1, 1))
    mixing_factors.uniform_()
    return mixing_factors


def calc_grad(x_hat, pred_hat):
    global grad_outputs
    if grad_outputs is None or pred_hat.size(0) != grad_outputs.size(0):
        grad_outputs = cudize(torch.ones(pred_hat.size()))
    return grad(outputs=pred_hat, inputs=x_hat, grad_outputs=grad_outputs, create_graph=True, retain_graph=True,
                only_inputs=True)[0]


def G_loss(G, D, fake_latents_in):
    G.zero_grad()
    z = Variable(fake_latents_in)
    g_ = G(z)
    d_fake, _ = D(g_)
    return -d_fake.mean()


def D_loss(D, G, real_images_in, fake_latents_in, loss_type, iwass_epsilon, grad_lambda, LAMBDA_2):
    D.zero_grad()
    G.zero_grad()
    x_real = Variable(real_images_in)
    d_real, d_last_real = D(x_real)
    with torch.no_grad():
        z = Variable(fake_latents_in)
    g_ = Variable(G(z).data)
    d_fake, _ = D(g_)
    if loss_type == 'hinge':
        d_loss = F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()
    else:
        if loss_type == 'wgan_theirs' or loss_type == 'wgan_theirs_ct':
            d_fake_mean = d_fake.mean()
            d_real_mean = d_real.mean()
            d_loss = d_fake_mean - d_real_mean + (d_fake_mean + d_real_mean) ** 2 * iwass_epsilon
            gp_gain = F.relu(d_real_mean-d_fake_mean)
        else:
            d_loss = d_fake.mean() - d_real.mean() + (d_real ** 2).mean() * iwass_epsilon
            gp_gain = 1
        if gp_gain != 0:
            alpha = get_mixing_factor(x_real.size(0))
            x_hat = Variable(alpha * x_real.data + (1.0 - alpha) * g_.data, requires_grad=True)
            pred_hat, _ = D(x_hat)
            g = calc_grad(x_hat, pred_hat).view(x_hat.size(0), -1)
            gp = g.norm(p=2, dim=1) - 1
            if loss_type == 'wgan_theirs' or loss_type == 'wgan_theirs_ct':
                gp = F.relu(gp)
            d_loss = d_loss + gp_gain * (gp ** 2).mean() * grad_lambda
        if (loss_type == 'wgan_ct' or loss_type == 'wgan_theirs_ct') and D.training:
            d_real2, d_last_real2 = D(x_real)
            CT = LAMBDA_2 * (d_real - d_real2) ** 2
            CT = CT + LAMBDA_2 * 0.1 * torch.mean((d_last_real - d_last_real2) ** 2, dim=1)
            d_loss = d_loss + torch.mean(torch.clamp(CT, min=0))
    return d_loss
