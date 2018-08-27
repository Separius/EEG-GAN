import torch
import random
from utils import cudize
import torch.nn.functional as F
from torch.autograd import Variable, grad

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


def G_loss(G, D, fake_latents_in, fake_latents_mixed, LAMBDA_3, random_multiply, loss_type):
    G.zero_grad()
    if not isinstance(fake_latents_in, (tuple, list)):
        fake_latents_in = (fake_latents_in,)
    z = (Variable(z) for z in fake_latents_in)
    g_ = G(*z)
    d_fake, _ = D(g_)
    scale = random.random() if random_multiply else 1.0
    # instead of Z_g||Z_t[0:T] use Z_g[0:T]||Z_t[0:T] => makes the global part useful
    if fake_latents_mixed is not None:
        z = (Variable(z) for z in fake_latents_mixed)
        g_ = G(*z)
        d_fake_mixed, _ = D(g_)
    if loss_type != 'dcgan':
        if fake_latents_mixed is None:
            return -d_fake.mean() * scale
        return (-d_fake.mean() + LAMBDA_3 * d_fake_mixed.mean()) * scale
    else:
        loss = F.binary_cross_entropy_with_logits(d_fake, cudize(Variable(torch.ones(d_fake.size(0)))))
        if fake_latents_mixed is None:
            return loss * scale
        return (loss + LAMBDA_3 * F.binary_cross_entropy_with_logits(d_fake, cudize(
            Variable(torch.zeros(d_fake_mixed.size(0)))))) * scale


def D_loss(D, G, real_images_in, fake_latents_in, concatenated_real, loss_type, iwass_epsilon, grad_lambda, LAMBDA_2,
           LAMBDA_3):
    D.zero_grad()
    G.zero_grad()
    x_real = Variable(real_images_in)
    d_real, d_last_real = D(x_real)
    if concatenated_real is not None:
        d_con, _ = D(concatenated_real)
    else:
        d_con = -1.0 if loss_type == 'hinge' else cudize(torch.zeros(1))
    with torch.no_grad():
        if not isinstance(fake_latents_in, (tuple, list)):
            fake_latents_in = (fake_latents_in,)
        z = (Variable(z) for z in fake_latents_in)
    g_ = Variable(G(*z).data)
    d_fake, _ = D(g_)
    if loss_type == 'hinge':
        d_loss = F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean() + F.relu(1.0 + d_con).mean() * LAMBDA_3
    elif loss_type == 'dcgan':
        x = torch.cat((d_real, d_fake), dim=0)
        y = torch.cat((cudize(Variable(torch.ones(d_real.size(0)))), cudize(Variable(torch.zeros(d_fake.size(0))))),
                      dim=0)
        d_loss = 2.0 * F.binary_cross_entropy_with_logits(x, y)
        if concatenated_real is not None and LAMBDA_3 != 0:
            d_loss = d_loss + F.binary_cross_entropy_with_logits(d_con, cudize(
                Variable(torch.zeros(d_con.size(0))))) * LAMBDA_3
    else:
        d_fake_mean = d_fake.mean()
        d_real_mean = d_real.mean()
        d_con_mean = d_con.mean() * LAMBDA_3
        if loss_type == 'wgan_theirs' or loss_type == 'wgan_theirs_ct':
            d_loss = d_fake_mean - d_real_mean + d_con_mean + (
                    d_fake_mean + d_real_mean + d_con_mean) ** 2 * iwass_epsilon
            gp_gain = F.relu(d_real_mean - d_fake_mean)
        else:
            d_loss = d_fake_mean - d_real_mean + d_con_mean * LAMBDA_3 + (d_real ** 2).mean() * iwass_epsilon
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
            CT = CT + LAMBDA_2 * 0.1 * torch.mean((d_last_real - d_last_real2) ** 2, dim=1).mean(dim=1)
            d_loss = d_loss + torch.mean(torch.clamp(CT, min=0))
    return d_loss
