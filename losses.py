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


def generator_loss(gen, dis, fake_latents_in, random_multiply):
    gen.zero_grad()
    if not isinstance(fake_latents_in, (tuple, list)):
        fake_latents_in = (fake_latents_in,)
    z = (Variable(z) for z in fake_latents_in)
    g_ = gen(*z)
    d_fake, _ = dis(g_)
    scale = random.random() if random_multiply else 1.0
    return -d_fake.mean() * scale


def discriminator_loss(dis, gen, real_images_in, fake_latents_in, loss_type, iwass_epsilon, grad_lambda, iwass_target):
    dis.zero_grad()
    gen.zero_grad()
    x_real = Variable(real_images_in)
    d_real, d_last_real = dis(x_real)
    with torch.no_grad():
        if not isinstance(fake_latents_in, (tuple, list)):
            fake_latents_in = (fake_latents_in,)
        z = [Variable(z) for z in fake_latents_in]
    g_ = Variable(gen(*z).data)
    d_fake, _ = dis(g_)
    if loss_type == 'hinge':
        d_loss = F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()
    else:
        d_fake_mean = d_fake.mean()
        d_real_mean = d_real.mean()
        if loss_type == 'wgan_theirs':
            d_loss = d_fake_mean - d_real_mean + (d_fake_mean + d_real_mean) ** 2 * iwass_epsilon
            gp_gain = F.relu(d_real_mean - d_fake_mean)
        else:
            d_loss = d_fake_mean - d_real_mean + (d_real ** 2).mean() * iwass_epsilon
            gp_gain = 1
        if gp_gain != 0:
            alpha = get_mixing_factor(x_real.size(0))
            min_size = min(g_.size(2), x_real.size(2))
            x_hat = Variable(alpha * x_real[:, :, :min_size].data + (1.0 - alpha) * g_[:, :, :min_size].data,
                             requires_grad=True)
            pred_hat, _ = dis(x_hat)
            g = calc_grad(x_hat, pred_hat).view(x_hat.size(0), -1)
            gp = g.norm(p=2, dim=1) - iwass_target
            if loss_type == 'wgan_theirs':
                gp = F.relu(gp)
            d_loss = d_loss + gp_gain * (gp ** 2).mean() * grad_lambda / (iwass_target ** 2)
    return d_loss
