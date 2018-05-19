from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from network import Generator, Discriminator
from wgan_gp_loss import G_loss, D_loss
from functools import partial
from trainer import Trainer
import dataset
from dataset import *
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from plugins import *
from utils import *
from functools import reduce
from argparse import ArgumentParser
from collections import OrderedDict

torch.manual_seed(1337)

default_params = OrderedDict(
    result_dir='results',
    exp_name='specs512',
    minibatch_size=16,
    lr_rampup_kimg=40,
    G_lr_max=0.001,
    D_lr_max=0.001,
    total_kimg=3000,
    tick_kimg_default=20,
    image_snapshot_ticks=3,
    resume_network='',
    resume_time=0,
    num_data_workers=2,
    random_seed=1337,
    progressive_growing=True,
    grad_lambda=10.0,
    iwass_epsilon=0.001,
    iwass_target=1.0,
    save_dataset='',
    load_dataset='',
    dataset_class='',
    postprocessors=[],
    checkpoints_dir='',
    loss_type='heuristic',
    label_smoothing=0.05,
    loss_of_mean=False,
    use_mixup=False,
    apply_sigmoid=False
)


class InfiniteRandomSampler(RandomSampler):

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


def main(params):
    if params['load_dataset']:
        dataset = load_pkl(params['load_dataset'])
    elif params['dataset_class']:
        dataset = globals()[params['dataset_class']](**params[params['dataset_class']])
        if params['save_dataset']:
            save_pkl(params['save_dataset'], dataset)
    else:
        raise Exception('One of either load_dataset (path to pkl) or dataset_class needs to be specified.')
    result_dir = create_result_subdir(params['result_dir'], params['exp_name'])

    losses = ['G_loss', 'D_loss', 'D_real', 'D_fake']
    stats_to_log = ['tick_stat', 'kimg_stat']
    if params['progressive_growing']:
        stats_to_log.extend(['depth', 'alpha', 'minibatch_size'])
    stats_to_log.extend(['time', 'sec.tick', 'sec.kimg'] + losses)
    logger = TeeLogger(os.path.join(result_dir, 'log.txt'), stats_to_log, [(1, 'epoch')])
    if params['resume_network']:
        G, D = load_models(params['resume_network'], params['result_dir'], logger)
    else:
        G = Generator(dataset.shape, **params['Generator'])
        D = Discriminator(dataset.shape, **params['Discriminator'])
    if params['progressive_growing']:
        assert G.max_depth == D.max_depth
    G = cudize(G)
    D = cudize(D)
    latent_size = params['Generator']['latent_size']
    logger.log('dataset shape: {}'.format(dataset.shape))
    logger.log(str(G))
    logger.log('Total number of parameters in Generator: {}'.format(
        sum(map(lambda x: reduce(lambda a, b: a * b, x.size()), G.parameters()))
    ))
    logger.log(str(D))
    logger.log('Total number of parameters in Discriminator: {}'.format(
        sum(map(lambda x: reduce(lambda a, b: a * b, x.size()), D.parameters()))
    ))

    def get_dataloader(minibatch_size):
        return DataLoader(dataset, minibatch_size, sampler=InfiniteRandomSampler(dataset),
                          num_workers=params['num_data_workers'], pin_memory=False, drop_last=True)

    def rl(bs):
        return lambda: random_latents(bs, latent_size)

    # Setting up learning rate and optimizers
    opt_g = Adam(G.parameters(), params['G_lr_max'], **params['Adam'])
    opt_d = Adam(D.parameters(), params['D_lr_max'], **params['Adam'])

    def rampup(cur_nimg):
        if cur_nimg < params['lr_rampup_kimg'] * 1000:
            p = max(0.0, 1 - cur_nimg / (params['lr_rampup_kimg'] * 1000))
            return np.exp(-p * p * 5.0)
        else:
            return 1.0

    lr_scheduler_d = LambdaLR(opt_d, rampup)
    lr_scheduler_g = LambdaLR(opt_g, rampup)

    mb_def = params['minibatch_size']
    D_loss_fun = partial(D_loss, params['loss_type'], params['iwass_epsilon'], params['iwass_target'],
                         params['grad_lambda'], params['label_smoothing'], params['loss_of_mean'],
                         params['use_mixup'], params['apply_sigmoid'])
    G_loss_fun = partial(G_loss, params['loss_type'], params['label_smoothing'], params['loss_of_mean'],
                         params['apply_sigmoid'])
    trainer = Trainer(D, G, D_loss_fun, G_loss_fun,
                      opt_d, opt_g, dataset, iter(get_dataloader(mb_def)), rl(mb_def), **params['Trainer'])
    # plugins
    if params['progressive_growing']:
        max_depth = min(G.max_depth, D.max_depth)
        trainer.register_plugin(DepthManager(get_dataloader, rl, max_depth, **params['DepthManager']))
    for i, loss_name in enumerate(losses):
        trainer.register_plugin(EfficientLossMonitor(i, loss_name))

    checkpoints_dir = params['checkpoints_dir'] if params['checkpoints_dir'] else result_dir
    trainer.register_plugin(SaverPlugin(checkpoints_dir, **params['SaverPlugin']))

    def subsitute_samples_path(d):
        return {k: (os.path.join(result_dir, v) if k == 'samples_path' else v) for k, v in d.items()}

    postprocessors = [globals()[x](**subsitute_samples_path(params[x])) for x in params['postprocessors']]
    trainer.register_plugin(OutputGenerator(lambda x: random_latents(x, latent_size),
                                            postprocessors, **params['OutputGenerator']))
    trainer.register_plugin(AbsoluteTimeMonitor(params['resume_time']))
    trainer.register_plugin(LRScheduler(lr_scheduler_d, lr_scheduler_g))
    trainer.register_plugin(logger)
    trainer.run(params['total_kimg'])
    dataset.close()


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.set_device(1)
    parser = ArgumentParser()
    needarg_classes = [Trainer, Generator, Discriminator, DepthManager, SaverPlugin, OutputGenerator, Adam]
    needarg_classes += get_all_classes(dataset)
    excludes = {'Adam': {'lr'}}
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
    params = get_structured_params(vars(parser.parse_args()))
    main(params)
