import os
import torch
import inspect
import numpy as np
from pickle import load, dump
from functools import partial
from argparse import ArgumentParser

EPSILON = 1e-8


def generate_samples(generator, gen_input):
    return generator.forward(gen_input).data.cpu().numpy()


def save_pkl(file_name, obj):
    with open(file_name, 'wb') as f:
        dump(obj, f)


def load_pkl(file_name):
    with open(file_name, 'rb') as f:
        return load(f)


def random_latents(num_latents, latent_size):
    return torch.randn(num_latents, latent_size)


def create_result_subdir(results_dir, experiment_name, dir_pattern='{new_num:03}-{exp_name}'):
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    fnames = os.listdir(results_dir)
    max_num = max(map(int, filter(lambda x: all(y.isdigit() for y in x), (x.split('-')[0] for x in fnames))), default=0)
    path = os.path.join(results_dir, dir_pattern.format(new_num=max_num + 1, exp_name=experiment_name))
    os.makedirs(path, exist_ok=False)
    return path


def num_params(net):
    model_parameters = trainable_params(net)
    return sum([np.prod(p.size()) for p in model_parameters])


def generic_arg_parse(x, hinttype=None):
    if hinttype is int or hinttype is float or hinttype is str:
        return hinttype(x)
    try:
        for _ in range(2):
            x = x.strip('\'').strip("\"")
        __special_tmp = eval(x, {}, {})
    except:  # the string contained some name - probably path, treat as string
        __special_tmp = x  # treat as string
    return __special_tmp


def create_params(classes, excludes=None, overrides=None):
    params = {}
    if not excludes:
        excludes = {}
    if not overrides:
        overrides = {}
    for cls in classes:
        nm = cls.__name__
        params[nm] = {
            k: (v.default if nm not in overrides or k not in overrides[nm] else overrides[nm][k])
            for k, v in dict(inspect.signature(cls.__init__).parameters).items()
            if v.default != inspect._empty and
               (nm not in excludes or k not in excludes[nm])
        }
    return params


def get_structured_params(params):
    new_params = {}
    for p in params:
        if '.' in p:
            [cls, attr] = p.split('.', 1)
            if cls not in new_params:
                new_params[cls] = {}
            new_params[cls][attr] = params[p]
        else:
            new_params[p] = params[p]
    return new_params


def cudize(thing):
    has_cuda = torch.cuda.is_available()
    if isinstance(thing, (list, tuple)):
        return [item.cuda() if has_cuda else item for item in thing]
    return thing.cuda() if has_cuda else thing


def trainable_params(model):
    return filter(lambda p: p.requires_grad, model.parameters())


def pixel_norm(h):
    mean = torch.mean(h * h, dim=1, keepdim=True)
    dom = torch.rsqrt(mean + EPSILON)
    return h * dom


def simple_argparser(default_params):
    parser = ArgumentParser()
    for k in default_params:
        parser.add_argument('--{}'.format(k), type=partial(generic_arg_parse, hinttype=type(default_params[k])))
    parser.set_defaults(**default_params)
    return get_structured_params(vars(parser.parse_args()))


def enable_benchmark():
    torch.backends.cudnn.benchmark = True  # for fast training(if network input size is almost constant)


def map_location(storage, location):
    return storage


def load_model(model_path):
    return torch.load(model_path, map_location=map_location)
