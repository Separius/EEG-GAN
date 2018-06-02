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


def merge_data(d_real, d_fake, label_smoothing):
    x = torch.cat((d_real, d_fake), dim=0)
    y = torch.cat((var(torch.ones(d_real.size(0)) - label_smoothing), var(torch.zeros(d_fake.size(0)))), dim=0)
    return x, y


def mixup(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = cudize(torch.randperm(batch_size))
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def d_loss_helper(x, y, criterion, use_mixup):
    if use_mixup:
        mixed_x, y_a, y_b, lam = mixup(x, y)
        d_loss = mixup_criterion(y_a, y_b, lam)(criterion, mixed_x)
    else:
        d_loss = criterion(x, y)
    return d_loss


def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def cross_entropy(pred, target, apply_sigmoid):
    if apply_sigmoid:
        return F.binary_cross_entropy(pred, target)
    return F.binary_cross_entropy_with_logits(pred, target)


def calc_grad(x_hat, pred_hat):
    global grad_outputs
    if grad_outputs is None or pred_hat.size(0) != grad_outputs.size(0):
        grad_outputs = cudize(torch.ones(pred_hat.size()))
    return grad(outputs=pred_hat, inputs=x_hat, grad_outputs=grad_outputs, create_graph=True, retain_graph=True,
                only_inputs=True)[0]


def per_disc_loss(d_fake, d_real, pred_hat, x_hat, loss_type, grad_lambda, label_smoothing, use_mixup,
                  apply_sigmoid, iwass_epsilon):
    if loss_type == 'square':
        x, y = merge_data(d_real, d_fake, label_smoothing)
        d_loss = d_loss_helper(x, y, lambda pred, target: torch.mean((pred - target) ** 2), use_mixup)
    elif loss_type in ['vanilla', 'heuristic']:
        x, y = merge_data(d_real, d_fake, label_smoothing)
        d_loss = d_loss_helper(x, y, lambda pred, target: 2.0 * cross_entropy(pred, target, apply_sigmoid), use_mixup)
    elif loss_type == 'dragan':
        x, y = merge_data(d_real, d_fake, label_smoothing)
        d_loss = d_loss_helper(x, y, lambda pred, target: 2.0 * cross_entropy(pred, target, apply_sigmoid), use_mixup)
        gradients = calc_grad(x_hat, pred_hat)
        gp = grad_lambda * ((gradients.norm(2, dim=1) - 1.0) ** 2).mean()
        d_loss = d_loss + gp
    elif loss_type == 'wgan_gp':
        d_loss = d_fake.mean() - d_real.mean() + (d_real ** 2).mean() * iwass_epsilon
        g = calc_grad(x_hat, pred_hat).view(x_hat.size(0), -1)
        gp = ((g.norm(p=2, dim=1) - 1) ** 2).mean() * grad_lambda
        d_loss = d_loss + gp
    elif loss_type == 'hinge':
        x, y = merge_data(d_real, d_fake, label_smoothing)
        y = 2 * y - 1.0
        d_loss = d_loss_helper(x, y, lambda pred, target: -2.0 * torch.mean(torch.clamp(x * y - 1.0, max=0.0)),
                               use_mixup)
    return d_loss.unsqueeze(0)


def disc_loss(d_fake, d_real, disc, real, fake, loss_type, iwass_epsilon, grad_lambda, label_smoothing, use_mixup,
              apply_sigmoid):
    if loss_type == 'dragan':
        alpha = get_mixing_factor(real.size(0))
        x_hat = Variable(
            alpha * real.data + (1.0 - alpha) * (real.data + 0.5 * real.data.std() * torch.rand(real.size())),
            requires_grad=True)
        pred_hat, _ = disc(x_hat)
    elif loss_type == 'wgan_gp':
        alpha = get_mixing_factor(real.size(0))
        x_hat = Variable(alpha * real.data + (1.0 - alpha) * fake.data, requires_grad=True)
        pred_hat, _ = disc(x_hat)
    else:
        x_hat = None
        pred_hat = None
    loss = per_disc_loss(d_fake, d_real, pred_hat, x_hat, loss_type, grad_lambda, label_smoothing, use_mixup,
                         apply_sigmoid, iwass_epsilon)
    return loss


def per_gen_loss(d_fake, loss_type, label_smoothing, apply_sigmoid):
    if loss_type == 'square':
        g_loss = torch.mean((d_fake - 1.0) ** 2) / 2.0
    elif loss_type == 'vanilla':
        g_loss = -cross_entropy(d_fake, var(torch.zeros(d_fake.size(0))), apply_sigmoid)
    elif loss_type == 'heuristic':
        g_loss = cross_entropy(d_fake, var(torch.ones(d_fake.size(0)) - label_smoothing), apply_sigmoid)
    elif loss_type == 'dragan':
        g_loss = cross_entropy(d_fake, var(torch.ones(d_fake.size(0)) - label_smoothing), apply_sigmoid)
    else:
        g_loss = -d_fake.mean()
    return g_loss.unsqueeze(0)


def G_loss(G, D, fake_latents_in, loss_type, label_smoothing, apply_sigmoid):
    G.zero_grad()
    z = Variable(fake_latents_in)
    g_ = G(z)
    d_fake, _ = D(g_)
    return per_gen_loss(d_fake, loss_type, label_smoothing, apply_sigmoid)


def D_loss(D, G, real_images_in, fake_latents_in, loss_type, iwass_epsilon, grad_lambda, label_smoothing, use_mixup,
           apply_sigmoid):
    D.zero_grad()
    G.zero_grad()
    x_real = Variable(real_images_in)
    d_real, _ = D(x_real)
    with torch.no_grad():
        z = Variable(fake_latents_in)
    g_ = Variable(G(z).data)
    d_fake, _ = D(g_)
    return disc_loss(d_fake, d_real, D, x_real, g_, loss_type, iwass_epsilon, grad_lambda, label_smoothing, use_mixup,
                     apply_sigmoid)
