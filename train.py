import os
import time
import yaml
import torch
import signal
import subprocess
import numpy as np
from box import Box
from trainer import Trainer
from torch.optim import Adam
from functools import partial
from dataset import EEGDataset
from losses import generator_loss, discriminator_loss
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from network import Generator, Discriminator
from torch.utils.data.sampler import SubsetRandomSampler
from plugins import (OutputGenerator, SaverPlugin, AbsoluteTimeMonitor,
                     EfficientLossMonitor, DepthManager, TeeLogger)
from utils import (load_pkl, save_pkl, cudize, random_latents, trainable_params, create_result_subdir,
                   num_params, create_params, generic_arg_parse, get_structured_params, enable_benchmark, load_model)

default_params = Box(
    result_dir='results',
    exp_name='',
    G_lr=0.001,
    total_kimg=6000,
    resume_network='',  # 001-test/network-snapshot-{}-000025.dat
    resume_time=0,
    num_data_workers=2,
    random_seed=1373,
    grad_lambda=10.0,  # must set it to zero to disable gp loss (even for non wgan based losses)
    iwass_epsilon=0.001,
    iwass_target=1.0,
    load_dataset='',
    loss_type='wgan_theirs',  # wgan_gp, hinge, wgan_theirs, rsgan, rasgan, rahinge
    cuda_device=0,
    ttur=False,
    config_file=None,
    fmap_base=2048,
    fmap_max=256,
    fmap_min=64,
    equalized=True,
    kernel_size=3,
    self_attention_layers=[],  # starts from 0 or null (for G it means putting it after ith layer)
    num_classes=0,
    monitor_threshold=10,
    monitor_warmup=50,
    monitor_patience=5,
    random_multiply=False,
    is_test=False
)


class InfiniteRandomSampler(SubsetRandomSampler):

    def __iter__(self):
        while True:
            it = super().__iter__()
            for x in it:
                yield x


def load_models(resume_network, result_dir, logger):
    logger.log('Resuming {}'.format(resume_network))
    G = load_model(os.path.join(result_dir, resume_network.format('generator')))
    D = load_model(os.path.join(result_dir, resume_network.format('discriminator')))
    return G, D


def thread_exit(_signal, frame):
    exit(0)


def worker_init(x):
    signal.signal(signal.SIGINT, thread_exit)


def test():
    from tqdm import tqdm
    from itertools import product
    fmap_base = 128
    fmap_max = 32
    fmap_min = 8
    dataset_shape = (64, 5, 128)
    latent_size = 32
    batch_size = 8
    losses = ['wgan_gp', 'hinge', 'wgan_theirs', 'rsgan', 'rasgan', 'rahinge']
    num_classes = [0]
    for initial_size, kernel_size, equalized, self_attention_layers, num_classes, sngan_rgb, act_alpha, residual, normalize_latents, spectral, no_tanh, act_norm, group_size, loss_type, grad_lambda, iwass_target, random_multiply in tqdm(
            product([2, 3], [3], [True], [[], [1, 2]], num_classes, [True, False], [0],
                    [True], [True], [True], [True, False], [None, 'layer', 'batch', 'pixel'],
                    [4, 8], losses, [0, 1], [1, 750], [True])):
        shared_model_params = dict(dataset_shape=dataset_shape, initial_size=initial_size, fmap_base=fmap_base,
                                   fmap_max=fmap_max, fmap_min=fmap_min, kernel_size=kernel_size, equalized=equalized,
                                   self_attention_layers=self_attention_layers, num_classes=num_classes)
        generator_params = dict(sngan_rgb=sngan_rgb, act_alpha=act_alpha, latent_size=latent_size, residual=residual,
                                normalize_latents=normalize_latents, dropout=0.1, do_mode='mul', spectral=spectral,
                                act_norm=act_norm, no_tanh=no_tanh)
        G = cudize(Generator(**shared_model_params, **generator_params))
        discriminator_params = dict(sngan_rgb=sngan_rgb, dropout=0.1, do_mode='mul', residual=residual,
                                    spectral=spectral, act_norm=act_norm, group_size=group_size, act_alpha=act_alpha)
        D = cudize(Discriminator(**shared_model_params, **discriminator_params))
        opt_g = Adam(trainable_params(G), 0.001)
        opt_d = Adam(trainable_params(D), 0.001)
        d_loss_fun = partial(discriminator_loss, loss_type=loss_type, iwass_epsilon=0.01, grad_lambda=grad_lambda,
                             iwass_target=iwass_target)
        g_loss_fun = partial(generator_loss, random_multiply=random_multiply, loss_type=loss_type)
        for gamma in [0, 0.5]:
            if gamma != 0 and len(self_attention_layers) == 0:
                continue
            G.set_gamma(gamma)
            D.set_gamma(gamma)
            for alpha in [0, 0.5, 1]:
                G.alpha = alpha
                D.alpha = alpha
                for depth in range(D.max_depth):
                    G.depth = depth
                    D.depth = depth
                    fake_latents_in = cudize(torch.randn(batch_size, latent_size))
                    real_images_expr = cudize(torch.randn(batch_size, 5, 2 ** (initial_size + depth)))
                    try:
                        d_loss = d_loss_fun(D, G, real_images_expr, fake_latents_in)
                        d_loss.backward()
                        opt_d.step()
                        g_loss = g_loss_fun(D, G, real_images_expr, fake_latents_in)
                        g_loss.backward()
                        opt_g.step()
                    except:
                        print(spectral, no_tanh, act_norm, loss_type, grad_lambda)
                        exit(1)


def main(params):
    dataset_params = params.EEGDataset
    if params.load_dataset and os.path.exists(params.load_dataset):
        print('loading dataset from file')
        dataset = load_pkl(params.load_dataset)
    else:
        print('creating dataset from scratch')
        dataset = EEGDataset(**dataset_params)
        if params.load_dataset:
            print('saving dataset to file')
            save_pkl(params.load_dataset, dataset)
    if params.config_file and params.exp_name == '':
        params.exp_name = params.config_file.split('/')[-1].split('.')[0]
    result_dir = create_result_subdir(params.result_dir, params.exp_name)

    losses = ['G_loss', 'D_loss']
    stats_to_log = ['tick_stat', 'kimg_stat']
    stats_to_log.extend(['depth', 'alpha', 'minibatch_size'])
    if len(params['self_attention_layers']) != 0:
        stats_to_log.extend(['gamma'])
    stats_to_log.extend(['time', 'sec.tick', 'sec.kimg'] + losses)

    logger = TeeLogger(os.path.join(result_dir, 'log.txt'), params.exp_name, stats_to_log, [(1, 'epoch')])
    if params.resume_network != '':
        G, D = load_models(params.resume_network, params.result_dir, logger)
    else:
        shared_model_params = dict(dataset_shape=dataset.shape, initial_size=dataset_params.model_dataset_depth_offset,
                                   fmap_base=params.fmap_base, fmap_max=params.fmap_max, fmap_min=params.fmap_min,
                                   kernel_size=params.kernel_size, equalized=params.equalized,
                                   self_attention_layers=params.self_attention_layers, num_classes=params.num_classes)
        G = Generator(**shared_model_params, **params.Generator)
        D = Discriminator(**shared_model_params, **params.Discriminator)
    latent_size = G.latent_size
    assert G.max_depth == D.max_depth
    G = cudize(G)
    D = cudize(D)
    d_loss_fun = partial(discriminator_loss, loss_type=params.loss_type, iwass_epsilon=params.iwass_epsilon,
                         grad_lambda=params.grad_lambda, iwass_target=params.iwass_target)
    g_loss_fun = partial(generator_loss, random_multiply=params.random_multiply, loss_type=params.loss_type)
    max_depth = G.max_depth

    logger.log('exp name: {}'.format(params.exp_name))
    try:
        logger.log('commit hash: {}'.format(subprocess.check_output(["git", "describe", "--always"]).strip()))
    except:
        logger.log('current time: {}'.format(time.time()))
    logger.log('dataset shape: {}'.format(dataset.shape))
    logger.log('Total number of parameters in Generator: {}'.format(num_params(G)))
    logger.log('Total number of parameters in Discriminator: {}'.format(num_params(D)))

    mb_def = params.DepthManager.minibatch_default
    dataset_len = len(dataset)
    train_idx = list(range(dataset_len))
    np.random.shuffle(train_idx)

    def get_dataloader(minibatch_size):
        return DataLoader(dataset, minibatch_size, sampler=InfiniteRandomSampler(train_idx), worker_init_fn=worker_init,
                          num_workers=params.num_data_workers, pin_memory=False, drop_last=True)

    def rl(bs):
        return partial(random_latents, num_latents=bs, latent_size=latent_size)

    def get_optimizers(g_lr):
        if params.ttur:
            d_lr = g_lr * 4.0
            params.Adam.betas = (0, 0.9)
        else:
            d_lr = g_lr
        opt_g = Adam(trainable_params(G), g_lr, **params.Adam)
        opt_d = Adam(trainable_params(D), d_lr, **params.Adam)
        return opt_g, opt_d

    opt_g, opt_d = get_optimizers(params.G_lr)

    trainer = Trainer(D, G, d_loss_fun, g_loss_fun, opt_d, opt_g, dataset, rl(mb_def), **params.Trainer)
    trainer.register_plugin(DepthManager(get_dataloader, rl, max_depth, params.Trainer.tick_kimg_default,
                                         len(params.self_attention_layers) != 0, get_optimizers, params.G_lr,
                                         **params.DepthManager))
    for i, loss_name in enumerate(losses):
        trainer.register_plugin(
            EfficientLossMonitor(i, loss_name, params.monitor_threshold, params.monitor_warmup,
                                 params.monitor_patience))

    trainer.register_plugin(SaverPlugin(result_dir, **params.SaverPlugin))
    trainer.register_plugin(
        OutputGenerator(lambda x: random_latents(x, latent_size), result_dir,
                        dataset_params.seq_len, dataset_params.dataset_freq, dataset_params.seq_len,
                        **params.OutputGenerator))
    trainer.register_plugin(AbsoluteTimeMonitor(params.resume_time))
    trainer.register_plugin(logger)
    yaml.dump(params, open(os.path.join(result_dir, 'conf.yml'), 'w'))
    trainer.run(params.total_kimg)
    del trainer


if __name__ == "__main__":
    parser = ArgumentParser()
    needarg_classes = [Trainer, Generator, Discriminator, DepthManager, SaverPlugin, OutputGenerator, Adam, EEGDataset]
    excludes = {'Adam': {'lr', 'amsgrad'}}
    default_overrides = {'Adam': {'betas': (0.0, 0.99)}}
    auto_args = create_params(needarg_classes, excludes, default_overrides)
    for k in default_params:
        parser.add_argument('--{}'.format(k), type=partial(generic_arg_parse, hinttype=type(default_params[k])))
    for cls in auto_args:
        group = parser.add_argument_group(cls, 'Arguments for initialization of class {}'.format(cls))
        for k in auto_args[cls]:
            name = '{}.{}'.format(cls, k)
            group.add_argument('--{}'.format(name), type=generic_arg_parse)
            default_params[name] = auto_args[cls][k]
    parser.set_defaults(**default_params)
    params = vars(parser.parse_args())
    if params['config_file']:
        print('loading config_file')
        params.update(yaml.load(open(params['config_file'], 'r')))
    params = Box(get_structured_params(params))
    np.random.seed(params.random_seed)
    torch.manual_seed(params.random_seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(params.cuda_device)
        torch.cuda.manual_seed_all(params.random_seed)
        enable_benchmark()
    if params['is_test']:
        test()
    else:
        main(params)
    print('training finished!')
