from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from network import Generator, Discriminator
from losses import G_loss, D_loss
from functools import partial
from trainer import Trainer
from dataset import MyDataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler, SubsetRandomSampler
from plugins import FixedNoise, OutputGenerator, Validator, GifGenerator, SaverPlugin, LRScheduler, \
    AbsoluteTimeMonitor, EfficientLossMonitor, DepthManager, TeeLogger
from utils import load_pkl, save_pkl, cudize, random_latents, trainable_params, create_result_subdir, num_params, \
    create_params, generic_arg_parse, get_structured_params
import numpy as np
import torch
import os
import signal
import yaml
import subprocess
from argparse import ArgumentParser
from collections import OrderedDict

default_params = OrderedDict(
    result_dir='results',
    exp_name='exp_name',
    lr_rampup_kimg=50,
    G_lr_max=0.001,
    D_lr_max=0.001,
    total_kimg=4000,
    resume_network='',
    resume_time=0,
    num_data_workers=2,
    random_seed=1337,
    grad_lambda=10.0,
    iwass_epsilon=0.001,
    save_dataset='',
    load_dataset='',
    loss_type='wgan_gp',  # wgan_gp, wgan_ct, hinge
    mixup_alpha=None,  # null or float (was 1.0 before)
    cuda_device=0,
    validation_split=0,
    LAMBDA_2=2,
    optimizer='adam',  # adam, amsgrad, ttur
    config_file=None
)


class InfiniteRandomSampler(SubsetRandomSampler):

    def __iter__(self):
        while True:
            it = super().__iter__()
            for x in it:
                yield x


def load_models(resume_network, result_dir, logger):
    logger.log('Resuming {}'.format(resume_network))
    G = torch.load(os.path.join(result_dir, resume_network.format('generator')))
    D = torch.load(os.path.join(result_dir, resume_network.format('discriminator')))
    return G, D


def thread_exit(_signal, frame):
    exit(0)


def worker_init(x):
    signal.signal(signal.SIGINT, thread_exit)


def main(params):
    if params['load_dataset'] and os.path.exists(params['load_dataset']):
        print('loading dataset from file')
        dataset = load_pkl(params['load_dataset'])
    else:
        print('loading dataset from scratch')
        dataset = MyDataset(**params['MyDataset'])
        if params['save_dataset'] or params['load_dataset']:
            print('saving dataset to file')
            save_pkl(params['save_dataset'] if params['save_dataset'] else params['load_dataset'], dataset)
    result_dir = create_result_subdir(params['result_dir'], params['exp_name'])

    losses = ['G_loss', 'D_loss']
    stats_to_log = ['tick_stat', 'kimg_stat']
    stats_to_log.extend(['depth', 'alpha', 'minibatch_size'])
    stats_to_log.extend(['time', 'sec.tick', 'sec.kimg'] + losses)
    num_channels = dataset.shape[1]
    if params['validation_split'] > 0:
        val_stats = ['d_loss']
        if num_channels != 1:
            for ch in range(num_channels):
                for cs in ['linear_svm', 'rbf_svm', 'decision_tree', 'random_forest']:
                    val_stats.append(cs + '_ch_' + str(ch))
        for cs in ['linear_svm', 'rbf_svm', 'decision_tree', 'random_forest']:
            val_stats.append(cs + '_all')
        stats_to_log.extend(['validation.' + x for x in val_stats])
    logger = TeeLogger(os.path.join(result_dir, 'log.txt'), stats_to_log, [(1, 'epoch')])

    if params['resume_network']:
        G, D = load_models(params['resume_network'], params['result_dir'], logger)
    else:
        G = Generator(dataset.shape, params['MyDataset']['model_dataset_depth_offset'], **params['Generator'])
        if params['Discriminator']['spectral_norm']:
            params['Discriminator']['normalization'] = None
        spectral_norm_linear = params['Discriminator']['spectral_norm']
        if params['loss_type'] != 'hinge':
            params['Discriminator']['spectral_norm'] = False
            if params['Discriminator']['normalization'] == 'batch_norm':
                params['Discriminator']['normalization'] = 'layer_norm'
        D = Discriminator(dataset.shape, params['MyDataset']['model_dataset_depth_offset'],
                          spectral_norm_linear=spectral_norm_linear, **params['Discriminator'])
    assert G.max_depth == D.max_depth
    G = cudize(G)
    D = cudize(D)
    latent_size = params['Generator']['latent_size']
    logger.log('commit hash: {}'.format(subprocess.check_output(["git", "describe", "--always"]).strip()))
    logger.log('dataset shape: {}'.format(dataset.shape))
    logger.log('Total number of parameters in Generator: {}'.format(num_params(G)))
    logger.log('Total number of parameters in Discriminator: {}'.format(num_params(D)))

    mb_def = params['DepthManager']['minibatch_default']
    dataset_len = len(dataset)
    indices = list(range(dataset_len))
    np.random.shuffle(indices)
    split = int(np.floor(params['validation_split'] * dataset_len))
    train_idx, valid_idx = indices[split:], indices[:split]
    valid_data_loader = DataLoader(dataset, batch_size=mb_def, sampler=SequentialSampler(valid_idx), drop_last=False,
                                   num_workers=params['num_data_workers'])

    def get_dataloader(minibatch_size):
        return DataLoader(dataset, minibatch_size, sampler=InfiniteRandomSampler(train_idx), worker_init_fn=worker_init,
                          num_workers=params['num_data_workers'], pin_memory=False, drop_last=True)

    def rl(bs):
        return lambda: random_latents(bs, latent_size)

    if params['optimizer'] == 'ttur':
        params['D_lr_max'] = params['G_lr_max'] * 4.0
    opt_g = Adam(trainable_params(G), params['G_lr_max'], amsgrad=params['optimizer'] == 'amsgrad', **params['Adam'])
    opt_d = Adam(trainable_params(D), params['D_lr_max'], amsgrad=params['optimizer'] == 'amsgrad', **params['Adam'])

    def rampup(cur_nimg):
        if cur_nimg < params['lr_rampup_kimg'] * 1000:
            p = max(0.0, 1 - cur_nimg / (params['lr_rampup_kimg'] * 1000))
            return np.exp(-p * p * 5.0)
        else:
            return 1.0

    lr_scheduler_d = LambdaLR(opt_d, rampup)
    lr_scheduler_g = LambdaLR(opt_g, rampup)

    D_loss_fun = partial(D_loss, loss_type=params['loss_type'], iwass_epsilon=params['iwass_epsilon'],
                         grad_lambda=params['grad_lambda'], mixup_alpha=params['mixup_alpha'],
                         LAMBDA_2=params['LAMBDA_2'])
    trainer = Trainer(D, G, D_loss_fun, G_loss, opt_d, opt_g, dataset, rl(mb_def), **params['Trainer'])
    max_depth = min(G.max_depth, D.max_depth)
    trainer.register_plugin(
        DepthManager(get_dataloader, rl, max_depth, params['Trainer']['tick_kimg_default'], **params['DepthManager']))
    for i, loss_name in enumerate(losses):
        trainer.register_plugin(EfficientLossMonitor(i, loss_name))

    trainer.register_plugin(SaverPlugin(result_dir, **params['SaverPlugin']))
    if params['validation_split'] > 0:
        trainer.register_plugin(
            Validator(lambda x: random_latents(x, latent_size), valid_data_loader, **params['Validator']))
    trainer.register_plugin(
        OutputGenerator(lambda x: random_latents(x, latent_size), result_dir, params['MyDataset']['seq_len'],
                        params['MyDataset']['max_freq'], params['MyDataset']['seq_len'], **params['OutputGenerator']))
    trainer.register_plugin(
        FixedNoise(lambda x: random_latents(x, latent_size), result_dir, params['MyDataset']['seq_len'],
                   params['MyDataset']['max_freq'], params['MyDataset']['seq_len'], **params['OutputGenerator']))
    trainer.register_plugin(
        GifGenerator(lambda x: random_latents(x, latent_size), result_dir, params['MyDataset']['seq_len'],
                     params['MyDataset']['max_freq'], params['OutputGenerator']['output_snapshot_ticks'],
                     params['MyDataset']['seq_len'], **params['GifGenerator']))
    trainer.register_plugin(AbsoluteTimeMonitor(params['resume_time']))
    trainer.register_plugin(LRScheduler(lr_scheduler_d, lr_scheduler_g))
    trainer.register_plugin(logger)
    yaml.dump(params, open(os.path.join(result_dir, 'conf.yml'), 'w'))
    trainer.run(params['total_kimg'])
    del trainer


if __name__ == "__main__":
    parser = ArgumentParser()
    needarg_classes = [Trainer, Generator, Discriminator, DepthManager, SaverPlugin,
                       OutputGenerator, Adam, GifGenerator, Validator, MyDataset]
    excludes = {'Adam': {'lr', 'amsgrad', 'weight_decay'}}
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
    params = get_structured_params(params)
    torch.manual_seed(params['random_seed'])
    if torch.cuda.is_available():
        torch.cuda.set_device(params['cuda_device'])
        torch.cuda.manual_seed_all(params['random_seed'])
    main(params)
