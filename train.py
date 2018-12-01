import os
import time
import yaml
import torch
import signal
import subprocess
import numpy as np
from trainer import Trainer
from torch.optim import Adam
from functools import partial
from dataset import EEGDataset
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from network import Generator, Discriminator
from torch.optim.lr_scheduler import LambdaLR
from losses import generator_loss, discriminator_loss
from torch.utils.data.sampler import SubsetRandomSampler
from plugins import (OutputGenerator, TeeLogger, AbsoluteTimeMonitor,
                     EfficientLossMonitor, DepthManager, SaverPlugin)
from utils import (cudize, random_latents, trainable_params, create_result_subdir, num_params,
                   create_params, generic_arg_parse, get_structured_params, enable_benchmark, load_model)

default_params = dict(
    result_dir='results',
    exp_name='',
    lr=0.001,  # generator's learning rate
    total_kimg=6000,
    resume_network='',  # 001-test/network-snapshot-{}-000025.dat
    num_data_workers=2,
    random_seed=1373,
    grad_lambda=10.0,  # must set it to zero to disable gp loss (even for non wgan based losses)
    iwass_drift_epsilon=0.001,
    iwass_target=1.0,
    feature_matching_lambda=0.0,
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
    random_multiply=False,
    lr_rampup_kimg=40,  # set to 0 to disable
)


class InfiniteRandomSampler(SubsetRandomSampler):
    def __iter__(self):
        while True:
            it = super().__iter__()
            for x in it:
                yield x


def load_models(resume_network, result_dir, logger):
    logger.log('Resuming {}'.format(resume_network))
    generator = load_model(os.path.join(result_dir, resume_network.format('generator')))
    discriminator = load_model(os.path.join(result_dir, resume_network.format('discriminator')))
    return generator, discriminator


def thread_exit(_signal, frame):
    exit(0)


def worker_init(x):
    signal.signal(signal.SIGINT, thread_exit)


def main(params):
    dataset_params = params['EEGDataset']
    dataset = EEGDataset.from_config(**dataset_params)
    if params['config_file'] and params['exp_name'] == '':
        params['exp_name'] = params['config_file'].split('/')[-1].split('.')[0]
    result_dir = create_result_subdir(params['result_dir'], params['exp_name'])

    losses = ['G_loss', 'D_loss']
    stats_to_log = ['tick_stat', 'kimg_stat']
    stats_to_log.extend(['depth', 'alpha', 'minibatch_size'])
    if len(params['self_attention_layers']) != 0:
        stats_to_log.extend(['gamma'])
    stats_to_log.extend(['time', 'sec.tick', 'sec.kimg'] + losses)

    logger = TeeLogger(os.path.join(result_dir, 'log.txt'), params['exp_name'], stats_to_log, [(1, 'epoch')])
    if params['resume_network'] != '':
        generator, discriminator = load_models(params['resume_network'], params['result_dir'], logger)
    else:
        shared_model_params = dict(dataset_shape=dataset.shape, initial_size=dataset.model_dataset_depth_offset,
                                   fmap_base=params['fmap_base'], fmap_max=params['fmap_max'],
                                   fmap_min=params['fmap_min'], kernel_size=params['kernel_size'],
                                   equalized=params['equalized'], self_attention_layers=params['self_attention_layers'],
                                   num_classes=params['num_classes'], progression_scale=dataset.progression_scale)
        generator = Generator(**shared_model_params, **params['Generator'])
        discriminator = Discriminator(**shared_model_params, **params['Discriminator'])
    latent_size = generator.latent_size
    assert generator.max_depth == discriminator.max_depth
    generator = cudize(generator)
    discriminator = cudize(discriminator)
    d_loss_fun = partial(discriminator_loss, loss_type=params['loss_type'],
                         iwass_drift_epsilon=params['iwass_drift_epsilon'], grad_lambda=params['grad_lambda'],
                         iwass_target=params['iwass_target'])
    g_loss_fun = partial(generator_loss, random_multiply=params['random_multiply'], loss_type=params['loss_type'],
                         feature_matching_lambda=params['feature_matching_lambda'])
    max_depth = generator.max_depth

    logger.log('exp name: {}'.format(params['exp_name']))
    try:
        logger.log('commit hash: {}'.format(subprocess.check_output(['git', 'describe', '--always']).strip()))
    except:
        logger.log('current time: {}'.format(time.time()))
    logger.log('dataset shape: {}'.format(dataset.shape))
    logger.log('Total number of parameters in Generator: {}'.format(num_params(generator)))
    logger.log('Total number of parameters in Discriminator: {}'.format(num_params(discriminator)))

    mb_def = params['DepthManager']['minibatch_default']
    dataset_len = len(dataset)
    train_idx = list(range(dataset_len))
    np.random.shuffle(train_idx)

    def get_dataloader(minibatch_size):
        return DataLoader(dataset, minibatch_size, sampler=InfiniteRandomSampler(train_idx), worker_init_fn=worker_init,
                          num_workers=params['num_data_workers'], pin_memory=False, drop_last=True)

    def get_random_latents(bs):
        return partial(random_latents, num_latents=bs, latent_size=latent_size)

    def rampup(cur_nimg):
        if cur_nimg < params['lr_rampup_kimg'] * 1000:
            p = max(0.0, 1 - cur_nimg / (params['lr_rampup_kimg'] * 1000))
            return np.exp(-p * p * 5.0)
        else:
            return 1.0

    def get_optimizers(g_lr, last_epoch=-1):
        if params['ttur']:
            d_lr = g_lr * 4.0
            params['Adam']['betas'] = (0, 0.9)
        else:
            d_lr = g_lr
        opt_g = Adam(trainable_params(generator), g_lr, **params['Adam'])
        opt_d = Adam(trainable_params(discriminator), d_lr, **params['Adam'])
        lr_scheduler_d = LambdaLR(opt_d, rampup, last_epoch)
        lr_scheduler_g = LambdaLR(opt_g, rampup, last_epoch)
        return opt_g, opt_d, lr_scheduler_g, lr_scheduler_d

    opt_g, opt_d, lr_scheduler_g, lr_scheduler_d = get_optimizers(params['lr'])
    trainer = Trainer(discriminator, generator, d_loss_fun, g_loss_fun, opt_d, opt_g, dataset,
                      get_random_latents(mb_def), lr_scheduler_g, lr_scheduler_d, **params['Trainer'])
    trainer.register_plugin(
        DepthManager(get_dataloader, get_random_latents, max_depth, params['Trainer']['tick_kimg_default'],
                     len(params['self_attention_layers']) != 0, get_optimizers, params['lr'],
                     **params['DepthManager']))
    for i, loss_name in enumerate(losses):
        trainer.register_plugin(EfficientLossMonitor(i, loss_name, **params['EfficientLossMonitor']))
    trainer.register_plugin(SaverPlugin(result_dir, **params.SaverPlugin))
    trainer.register_plugin(
        OutputGenerator(lambda x: random_latents(x, latent_size), result_dir, dataset.seq_len, dataset.dataset_freq,
                        dataset.seq_len, **params['OutputGenerator']))
    trainer.register_plugin(AbsoluteTimeMonitor())
    trainer.register_plugin(logger)
    yaml.dump(params, open(os.path.join(result_dir, 'conf.yml'), 'w'))
    trainer.run(params['total_kimg'])
    del trainer


if __name__ == "__main__":
    parser = ArgumentParser()
    need_arg_classes = [Trainer, Generator, Discriminator, DepthManager, SaverPlugin,
                        OutputGenerator, Adam, EEGDataset, EfficientLossMonitor]
    excludes = {'Adam': {'lr', 'amsgrad'}}
    default_overrides = {'Adam': {'betas': (0.0, 0.99)}}
    auto_args = create_params(need_arg_classes, excludes, default_overrides)
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
    params = get_structured_params(params)
    np.random.seed(params['random_seed'])
    torch.manual_seed(params['random_seed'])
    if torch.cuda.is_available():
        torch.cuda.set_device(params['cuda_device'])
        torch.cuda.manual_seed_all(params['random_seed'])
        enable_benchmark()
    main(params)
    print('training finished!')
