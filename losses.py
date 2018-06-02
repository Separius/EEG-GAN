import torch
from torch.autograd import Variable, grad
from utils import var, cudize, ll
import numpy as np
import torch.nn.functional as F

mixing_factors = None
grad_outputs = None


def mul_rowwise(a, b):
    s = a.size()
    return (a.view(s[0], -1) * b).view(s)


def calc_gradient_penalty(D, real_data, fake_data, iwass_lambda, iwass_target):
    global mixing_factors, grad_outputs
    if mixing_factors is None or real_data.size(0) != mixing_factors.size(0):
        if torch.cuda.is_available():
            mixing_factors = torch.cuda.FloatTensor(real_data.size(0), 1)
        else:
            mixing_factors = torch.FloatTensor(real_data.size(0), 1)
    mixing_factors.uniform_()
    mixed_data = Variable(mul_rowwise(real_data, 1 - mixing_factors) + mul_rowwise(fake_data, mixing_factors),
                          requires_grad=True)
    mixed_scores = D(mixed_data)
    if grad_outputs is None or mixed_scores.size(0) != grad_outputs.size(0):
        if torch.cuda.is_available():
            grad_outputs = torch.cuda.FloatTensor(mixed_scores.size())
        else:
            grad_outputs = torch.FloatTensor(mixed_scores.size())
        grad_outputs.fill_(1.)
    gradients = grad(outputs=mixed_scores, inputs=mixed_data,
                     grad_outputs=grad_outputs,
                     create_graph=True, retain_graph=True,
                     only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - iwass_target) ** 2) * iwass_lambda / (iwass_target ** 2)
    return gradient_penalty


def mean_time_series(x, loss_of_mean):
    if loss_of_mean:
        x = x.mean(dim=-1)
    return x.view(-1)


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


def per_disc_loss(d_fake, d_real, pred_hat, x_hat, loss_type, grad_lambda, label_smoothing, loss_of_mean, use_mixup,
                  apply_sigmoid, iwass_epsilon):
    d_fake = mean_time_series(d_fake, loss_of_mean)
    d_real = mean_time_series(d_real, loss_of_mean)
    if loss_type == 'square':
        x, y = merge_data(d_real, d_fake, label_smoothing)
        d_loss = d_loss_helper(x, y, lambda pred, target: torch.mean((pred - target) ** 2), use_mixup)
    elif loss_type in ['vanilla', 'heuristic']:
        x, y = merge_data(d_real, d_fake, label_smoothing)
        d_loss = d_loss_helper(x, y, lambda pred, target: 2.0 * cross_entropy(pred, target, apply_sigmoid), use_mixup)
    elif loss_type == 'dragan':
        x, y = merge_data(d_real, d_fake, label_smoothing)
        d_loss = cross_entropy(x, y, apply_sigmoid) * 2.0
        pred_hat = mean_time_series(pred_hat, loss_of_mean)
        gradients = grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()), create_graph=True,
                         retain_graph=True, only_inputs=True)[0]
        gp = grad_lambda * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        d_loss = d_loss + gp
    elif loss_type == 'wgan_gp':
        d_loss = d_fake.mean() - d_real.mean()
        pred_hat = mean_time_series(pred_hat, loss_of_mean)
        g = grad(pred_hat, x_hat, grad_outputs=cudize(torch.ones(pred_hat.size())), create_graph=True)[0].view(
            x_hat.size(0), -1)
        gp = ((g.norm(p=2, dim=1) - 1) ** 2).mean() * grad_lambda
        d_loss = d_loss + gp
    else:
        D_real_loss = -d_real + d_real ** 2 * iwass_epsilon
        D_fake_loss = d_fake
        d_loss = (D_fake_loss + D_real_loss).mean()
    return d_loss.unsqueeze(0)


def disc_loss(d_fake, d_real, disc, real, fake, loss_type, iwass_epsilon, iwass_target, grad_lambda, label_smoothing,
              loss_of_mean, use_mixup, apply_sigmoid):
    if loss_type == 'dragan':
        alpha = cudize(torch.rand(real.size(0), 1, 1).expand(real.size()))
        x_hat = Variable(
            alpha * real.data + (1.0 - alpha) * (real.data + 0.5 * real.data.std() * torch.rand(real.size())),
            requires_grad=True)
        pred_hat = disc(x_hat)
    elif loss_type == 'wgan_gp':
        alpha = Variable(cudize(torch.rand(real.size(0), 1, 1).expand(real.size())), requires_grad=True)
        x_hat = real + alpha * (fake - real)
        pred_hat = disc(x_hat)
    else:
        x_hat = None
        pred_hat = None
    loss = per_disc_loss(d_fake, d_real, pred_hat, x_hat, loss_type, grad_lambda, label_smoothing, loss_of_mean, use_mixup, apply_sigmoid, iwass_epsilon)
    if loss_type == 'wgan_theirs':
        given_gp = calc_gradient_penalty(disc, real.data, fake.data, grad_lambda, iwass_target)
        loss = given_gp.mean() + loss
    return loss

def per_gen_loss(d_fake, loss_type, label_smoothing, loss_of_mean, apply_sigmoid):
    d_fake = mean_time_series(d_fake, loss_of_mean)
    if loss_type == 'square':
        g_loss = torch.mean((d_fake - 1.0) ** 2) / 2.0
    elif loss_type == 'vanilla':
        g_loss = -cross_entropy(d_fake, var(torch.zeros(d_fake.size(0))), apply_sigmoid)
    elif loss_type == 'heuristic':
        g_loss = cross_entropy(d_fake, var(torch.ones(d_fake.size(0)) - label_smoothing), apply_sigmoid)
    elif loss_type == 'dragan':
        g_loss = cross_entropy(d_fake, var(torch.ones(d_fake.size(0))), apply_sigmoid)
    else:
        g_loss = -d_fake.mean()
    return g_loss.unsqueeze(0)


def G_loss(G, D, fake_latents_in, loss_type, label_smoothing, loss_of_mean, apply_sigmoid):
    G.zero_grad()
    z = Variable(fake_latents_in)
    g_ = G(z)
    d_fake = D(g_)
    return per_gen_loss(d_fake, loss_type, label_smoothing, loss_of_mean, apply_sigmoid)


def D_loss(D, G, real_images_in, fake_latents_in, loss_type, iwass_epsilon, iwass_target, grad_lambda, label_smoothing,
           loss_of_mean, use_mixup, apply_sigmoid):
    D.zero_grad()
    G.zero_grad()
    x_real = Variable(real_images_in)
    d_real = D(x_real)
    with torch.no_grad():
        z = Variable(fake_latents_in)
    g_ = Variable(G(z).data)
    d_fake = D(g_)
    return disc_loss(d_fake, d_real, D, x_real, g_, loss_type, iwass_epsilon, iwass_target, grad_lambda,
                     label_smoothing, loss_of_mean, use_mixup, apply_sigmoid)
