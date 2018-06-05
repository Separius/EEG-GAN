import torch
from torch.autograd import Variable, grad
from utils import var, cudize
import numpy as np
import torch.nn.functional as F

mixing_factors = None
grad_outputs = None


def get_mixing_factor(batch_size):
    global mixing_factors
    if mixing_factors is None or batch_size != mixing_factors.size(0):
        mixing_factors = cudize(torch.FloatTensor(batch_size, 1, 1))
    mixing_factors.uniform_()
    return mixing_factors


def merge_data(d_real, d_fake):
    x = torch.cat((d_real, d_fake), dim=0)
    y = torch.cat((var(torch.ones(d_real.size(0))), var(-torch.ones(d_fake.size(0)))), dim=0)
    return x, y


def mixup(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = cudize(torch.randperm(batch_size))
    mixed_x = lam * x + (1.0 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def d_loss_helper(x, y, criterion, mixup_alpha=1.0):
    if mixup_alpha is not None:
        mixed_x, y_a, y_b, lam = mixup(x, y, mixup_alpha)
        d_loss = mixup_criterion(y_a, y_b, lam)(criterion, mixed_x)
    else:
        d_loss = criterion(x, y)
    return d_loss


def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


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


def D_loss(D, G, real_images_in, fake_latents_in, loss_type, iwass_epsilon, grad_lambda, LAMBDA_2, mixup_alpha):
    D.zero_grad()
    G.zero_grad()
    x_real = Variable(real_images_in)
    d_real, d_last_real = D(x_real)
    with torch.no_grad():
        z = Variable(fake_latents_in)
    g_ = Variable(G(z).data)
    d_fake, _ = D(g_)
    if loss_type == 'wgan_gp' or loss_type == 'wgan_ct':
        alpha = get_mixing_factor(x_real.size(0))
        x_hat = Variable(alpha * x_real.data + (1.0 - alpha) * g_.data, requires_grad=True)
        pred_hat, _ = D(x_hat)
        d_loss = d_loss_helper(*merge_data(d_real, d_fake), lambda pred, label: -2.0 * (pred * label).mean(),
                               mixup_alpha) + (d_real ** 2).mean() * iwass_epsilon
        # d_loss = d_fake.mean() - d_real.mean() + (d_real ** 2).mean() * iwass_epsilon
        g = calc_grad(x_hat, pred_hat).view(x_hat.size(0), -1)
        gp = ((g.norm(p=2, dim=1) - 1) ** 2).mean() * grad_lambda
        d_loss = d_loss + gp
        if loss_type == 'wgan_ct' and D.training:
            d_real2, d_last_real2 = D(x_real)
            CT = LAMBDA_2 * (d_real - d_real2) ** 2
            CT = CT + LAMBDA_2 * 0.1 * torch.mean((d_last_real - d_last_real2) ** 2, dim=1)
            d_loss = d_loss + torch.mean(torch.clamp(CT, min=0))
    else:
        d_loss = d_loss_helper(*merge_data(d_real, d_fake),
                               lambda pred, label: -2.0 * torch.clamp(-1.0 + label * pred, max=0.0).mean(), mixup_alpha)
        # d_loss = -torch.clamp(-1.0+d_real, max=0.0).mean()-torch.clamp(-1-d_fake, max=0.0).mean()
    return d_loss
