import os
import signal
import subprocess
import time
from functools import partial

import numpy as np
import torch
import yaml
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from dataset import EEGDataset, get_collate_real, get_collate_fake
from losses import generator_loss, discriminator_loss
from network import Generator, Discriminator
from plugins import (OutputGenerator, TeeLogger, AbsoluteTimeMonitor, SlicedWDistance, CheckCond,
                     EfficientLossMonitor, DepthManager, SaverPlugin, EvalDiscriminator, WatchSingularValues)
from trainer import Trainer
from utils import cudize, random_latents, trainable_params, create_result_subdir, num_params, parse_config, load_model

default_params = dict(
    result_dir='results',
    exp_name='',
    lr=0.001,  # generator's learning rate
    total_kimg=6000,
    resume_network='',  # 001-test/network-snapshot-{}-000025.dat
    num_data_workers=0,
    random_seed=1373,
    grad_lambda=10.0,  # must set it to zero to disable gp loss (even for non wgan based losses)
    iwass_drift_epsilon=0.001,
    iwass_target=1.0,
    feature_matching_lambda=0.0,
    loss_type='wgan_gp',  # wgan_gp, hinge, wgan_theirs, rsgan, rasgan, rahinge
    cuda_device=0,
    ttur=False,
    config_file=None,
    fmap_base=1024,
    fmap_max=256,
    fmap_min=64,
    equalized=True,
    kernel_size=3,
    self_attention_layers=[],  # starts from 0 or null (for G it means putting it after ith layer)
    random_multiply=False,
    lr_rampup_kimg=0.0,  # set to 0 to disable (used to be 40)
    z_distribution='normal',  # or 'bernoulli' or 'censored'
    init='kaiming_normal',  # or xavier_uniform or orthogonal
    act_alpha=0.2,
    residual=False,
    sagan_non_local=True,
    use_factorized_attention=False,
    calc_swd=False,
    correlation_pairs=[],  # [[0, 1], [1, 2], [2, 3]], #global condition
    bands=[]  # [0.5, 4, 8, 12, 30]  # , 100] #temporal condition
)


class InfiniteRandomSampler(SubsetRandomSampler):
    def __iter__(self):
        while True:
            it = super().__iter__()
            for x in it:
                yield x


def load_models(resume_network, result_dir, logger):
    logger.log('Resuming {}'.format(resume_network))
    dest = os.path.join(result_dir, resume_network)
    generator, g_optimizer, g_cur_img = load_model(dest.format('generator'), True)
    discriminator, d_optimizer, d_cur_img = load_model(dest.format('discriminator'), True)
    assert g_cur_img == d_cur_img
    return generator, g_optimizer, discriminator, d_optimizer, g_cur_img


def thread_exit(_signal, frame):
    exit(0)


def worker_init(x):
    signal.signal(signal.SIGINT, thread_exit)


def main(params):
    dataset_params = params['EEGDataset']
    dataset, val_dataset = EEGDataset.from_config(**dataset_params)
    if params['config_file'] and params['exp_name'] == '':
        params['exp_name'] = params['config_file'].split('/')[-1].split('.')[0]
    result_dir = create_result_subdir(params['result_dir'], params['exp_name'])

    losses = ['G_loss', 'D_loss']
    total_temporal_conds = ((len(params['bands']) - 1) * dataset.num_channels) if len(params['bands']) > 1 else 0
    global_conds = len(params['correlation_pairs'])
    stats_to_log = ['tick_stat', 'kimg_stat']
    stats_to_log.extend(['depth', 'alpha', 'minibatch_size'])
    if len(params['self_attention_layers']) != 0:
        stats_to_log.extend(['gamma'])
    stats_to_log.extend(['time', 'sec.tick', 'sec.kimg'] + losses)
    if dataset_params['validation_ratio'] > 0:
        stats_to_log.extend(['memorization.val', 'memorization.epoch'])
    if params['calc_swd']:
        stats_to_log.extend(['swd.val', 'swd.epoch'])
    if global_conds > 0:
        stats_to_log.extend(['cond.global_distance'])
    if total_temporal_conds > 0:
        stats_to_log.extend(['cond.temporal_distance'])

    logger = TeeLogger(os.path.join(result_dir, 'log.txt'), params['exp_name'], stats_to_log, [(1, 'epoch')])
    shared_model_params = dict(initial_kernel_size=dataset.initial_kernel_size, num_channels=dataset.num_channels,
                               fmap_base=params['fmap_base'], fmap_max=params['fmap_max'], fmap_min=params['fmap_min'],
                               kernel_size=params['kernel_size'], equalized=params['equalized'],
                               self_attention_layers=params['self_attention_layers'],
                               progression_scale_up=dataset.progression_scale_up,
                               progression_scale_down=dataset.progression_scale_down, init=params['init'],
                               act_alpha=params['act_alpha'], residual=params['residual'],
                               sagan_non_local=params['sagan_non_local'],
                               factorized_attention=params['use_factorized_attention'])
    for n in ('Generator', 'Discriminator'):
        p = params[n]
        if p['spectral']:
            if p['act_norm'] == 'pixel':
                logger.log('Warning, setting pixel normalization with spectral norm in {} is not a good idea'.format(n))
            if params['equalized']:
                logger.log('Warning, setting equalized weights with spectral norm in {} is not a good idea'.format(n))
    if params['DepthManager']['disable_progression'] and not params['residual']:
        logger.log('Warning, you have set the residual to false and disabled the progression')
    if params['Discriminator']['act_norm'] is not None:
        logger.log('Warning, you are using an activation normalization in discriminator')
    temporal_conds = {'temporal_1': total_temporal_conds} if len(params['bands']) > 1 else {}
    generator = Generator(**shared_model_params, z_distribution=params['z_distribution'],
                          num_conds=global_conds + total_temporal_conds, **params['Generator'])
    discriminator = Discriminator(**shared_model_params, global_conds=global_conds, temporal_conds=temporal_conds,
                                  **params['Discriminator'])

    def rampup(cur_nimg):
        if cur_nimg < params['lr_rampup_kimg'] * 1000:
            p = max(0.0, 1 - cur_nimg / (params['lr_rampup_kimg'] * 1000))
            return np.exp(-p * p * 5.0)
        else:
            return 1.0

    if params['ttur']:
        params['Adam']['betas'] = (0, 0.9)

    def get_optimizers(g_lr):
        d_lr = g_lr
        if params['ttur']:
            d_lr *= 4.0
        opt_g = Adam(trainable_params(generator), g_lr, **params['Adam'])
        opt_d = Adam(trainable_params(discriminator), d_lr, **params['Adam'])
        if params['lr_rampup_kimg'] > 0:
            lr_scheduler_g = LambdaLR(opt_g, rampup, -1)
            lr_scheduler_d = LambdaLR(opt_d, rampup, -1)
            return opt_g, opt_d, lr_scheduler_g, lr_scheduler_d
        return opt_g, opt_d, None, None

    if params['resume_network'] != '':
        logger.log('resuming networks')
        generator_state, opt_g_state, discriminator_state, opt_d_state, train_cur_img = load_models(
            params['resume_network'], params['result_dir'], logger)
        generator.load_state_dict(generator_state)
        discriminator.load_state_dict(discriminator_state)
        opt_g, opt_d, _, _ = get_optimizers(params['lr'])
        opt_g.load_state_dict(opt_g_state)
        opt_d.load_state_dict(opt_d_state)
    else:
        opt_g = None
        opt_d = None
        train_cur_img = 0
    latent_size = generator.input_latent_size
    generator.train()
    discriminator.train()
    generator = cudize(generator)
    discriminator = cudize(discriminator)
    if opt_g is not None:
        for opt in [opt_g, opt_d]:
            for state in opt.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = cudize(v)
    d_loss_fun = partial(discriminator_loss, loss_type=params['loss_type'], iwass_target=params['iwass_target'],
                         iwass_drift_epsilon=params['iwass_drift_epsilon'], grad_lambda=params['grad_lambda'])
    g_loss_fun = partial(generator_loss, random_multiply=params['random_multiply'], loss_type=params['loss_type'],
                         feature_matching_lambda=params['feature_matching_lambda'])
    max_depth = generator.max_depth

    logger.log('exp name: {}'.format(params['exp_name']))
    try:
        logger.log('commit hash: {}'.format(subprocess.check_output(['git', 'describe', '--always']).strip()))
    except:
        logger.log('current time: {}'.format(time.time()))
    logger.log('training dataset shape: {}'.format(dataset.shape))
    if dataset_params['validation_ratio'] > 0:
        logger.log('val dataset shape: {}'.format(val_dataset.shape))
    logger.log('Total number of parameters in Generator: {}'.format(num_params(generator)))
    logger.log('Total number of parameters in Discriminator: {}'.format(num_params(discriminator)))

    mb_def = params['DepthManager']['minibatch_default']

    collate_real = get_collate_real(dataset.end_sampling_freq, dataset.seq_len,
                                    params['bands'], params['correlation_pairs'])
    collate_fake = get_collate_fake(latent_size, params['z_distribution'], collate_real)

    def get_dataloader(minibatch_size, is_training=True, depth=0, alpha=1, is_real=True):
        ds = dataset if is_training else val_dataset
        shared_dataloader_params = {'dataset': ds, 'batch_size': minibatch_size, 'drop_last': True,
                                    'worker_init_fn': worker_init, 'num_workers': params['num_data_workers'],
                                    'collate_fn': collate_real if is_real else collate_fake}
        if not is_training:
            ds.model_depth = depth
            ds.alpha = alpha
            # NOTE you must drop last in order to be compatible with D.stats layer
            return DataLoader(**shared_dataloader_params, shuffle=True)
        return DataLoader(**shared_dataloader_params, sampler=InfiniteRandomSampler(list(range(len(ds)))))

    # NOTE you can not put the if inside your function (a function should either return or yield)
    if global_conds + total_temporal_conds > 0:
        def get_random_latents(minibatch_size, is_training=True, depth=0, alpha=1):
            return iter(get_dataloader(minibatch_size, is_training, depth, alpha, False))
    else:
        def get_random_latents(minibatch_size, is_training=True, depth=0, alpha=1):
            while True:
                yield {'z': cudize(random_latents(minibatch_size, latent_size, params['z_distribution']))}

    trainer = Trainer(discriminator, generator, d_loss_fun, g_loss_fun, dataset, get_random_latents(mb_def),
                      train_cur_img, opt_g, opt_d, **params['Trainer'])
    trainer.register_plugin(
        DepthManager(get_dataloader, get_random_latents, max_depth, params['Trainer']['tick_kimg_default'],
                     len(params['self_attention_layers']) != 0, get_optimizers, params['lr'],
                     **params['DepthManager']))
    for i, loss_name in enumerate(losses):
        trainer.register_plugin(EfficientLossMonitor(i, loss_name, **params['EfficientLossMonitor']))
    trainer.register_plugin(SaverPlugin(result_dir, **params['SaverPlugin']))
    trainer.register_plugin(
        OutputGenerator(lambda x: get_random_latents(x), result_dir, dataset.seq_len, dataset.end_sampling_freq,
                        dataset.seq_len, **params['OutputGenerator']))
    if dataset_params['validation_ratio'] > 0:
        trainer.register_plugin(EvalDiscriminator(get_dataloader, params['SaverPlugin']['network_snapshot_ticks']))
    if params['calc_swd']:
        trainer.register_plugin(
            SlicedWDistance(dataset.progression_scale, params['SaverPlugin']['network_snapshot_ticks'],
                            **params['SlicedWDistance']))
    if total_temporal_conds + global_conds > 0:
        trainer.register_plugin(CheckCond(get_random_latents, dataset.end_sampling_freq, dataset.seq_len,
                                          params['bands'], params['correlation_pairs']))
    trainer.register_plugin(AbsoluteTimeMonitor())
    if params['Generator']['spectral']:
        trainer.register_plugin(WatchSingularValues(generator, **params['WatchSingularValues']))
    if params['Discriminator']['spectral']:
        trainer.register_plugin(WatchSingularValues(discriminator, **params['WatchSingularValues']))
    trainer.register_plugin(logger)
    yaml.dump(params, open(os.path.join(result_dir, 'conf.yml'), 'w'))
    trainer.run(params['total_kimg'])
    del trainer


if __name__ == "__main__":
    need_arg_classes = [Trainer, Generator, Discriminator, Adam, OutputGenerator, DepthManager, SaverPlugin,
                        SlicedWDistance, EfficientLossMonitor, EvalDiscriminator, EEGDataset, WatchSingularValues]
    main(parse_config(default_params, need_arg_classes))
    print('training finished!')
