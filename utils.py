import torch
import os
import inspect
from pickle import load, dump
from pyeeg import spectral_entropy_vectorized, bin_power_vectorized, hjorth_vectorized
import numpy as np
from scipy.stats import kurtosis, skew
from sklearn import svm, tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from argparse import ArgumentParser
from functools import partial


def generate_samples(generator, gen_input):
    out = generator.forward(gen_input)
    out = out.cpu().data.numpy()
    return out


def save_pkl(fname, obj):
    with open(fname, 'wb') as f:
        dump(obj, f)


def load_pkl(fname):
    with open(fname, 'rb') as f:
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
    return thing.cuda() if torch.cuda.is_available() else thing


def ll(loss):
    return loss.data.cpu().item()


def trainable_params(model):
    return filter(lambda p: p.requires_grad, model.parameters())


def zcc(x):
    return ((x[:, :-1] * x[:, 1:]) < 0).sum(axis=1)


def get_accuracy(real, fake):
    y = np.concatenate((np.ones(real[0].shape[0]), np.zeros(fake[0].shape[0])), axis=0)
    result = dict()
    if len(real) != 1:
        means = []
        for i, (real_ch, fake_ch) in enumerate(zip(real, fake)):
            for v in evaluate_classifier(np.concatenate((real_ch, fake_ch), axis=0), y, '_ch_{}'.format(i)).values():
                means.append(v)
        result['ch_acc'] = sum(means) / len(means)
    means = []
    for v in evaluate_classifier(np.concatenate((np.concatenate(real, axis=1), np.concatenate(fake, axis=1)), axis=0),
                                 y, '_all').values():
        means.append(v)
    result['mul_acc'] = sum(means) / len(means)
    return result


def evaluate_classifier(x, y, postfix, cross_validation=5):
    return {name + postfix: cross_val_score(clf, x, y, cv=cross_validation).mean() for name, clf in
            zip(['linear_svm', 'rbf_svm', 'decision_tree', 'random_forest'],
                [svm.SVC(kernel='linear', C=1), svm.SVC(kernel='rbf', C=1),
                 tree.DecisionTreeClassifier(max_depth=5, max_features='sqrt', presort=True),
                 RandomForestClassifier(max_depth=5, max_features='sqrt')])}


def get_features_vectorized(x, is_numpy=False):
    z = x if is_numpy else x.cpu().data.numpy()
    z = z.reshape((-1, z.shape[-1]))
    if not np.isfinite(z).all():
        print('INPUT IS NOT FINITE!!')
        exit()
    res = np.zeros((z.shape[0], 21))
    N = z.shape[1]
    diff = np.diff(z)
    res[:, 0] = zcc(diff)
    res[:, 1] = np.log10(N) / (np.log10(N) + np.log10(N / (N + 0.4 * res[:, 0])))
    res[:, 2] = np.mean(z, axis=1)
    res[:, 3] = np.std(z, axis=1)
    res[:, 4] = np.sqrt(np.mean(np.square(z), axis=1))
    res[:, 5] = np.mean(diff, axis=1)
    res[:, 6] = np.mean(np.abs(diff), axis=1)
    res[:, 7] = np.mean(np.abs(diff) < 0.005, axis=1)
    res[:, 8] = kurtosis(z, axis=1)
    res[:, 9] = skew(z, axis=1)
    res[:, 10] = zcc(z)
    fourier = np.fft.fft(z, axis=1)
    frequencies = np.fft.fftfreq(N, 1 / 80.0)
    magnitudes = np.abs(fourier[:, np.where(frequencies >= 0)[0]])
    res[:, 11] = np.sum(magnitudes * frequencies[np.where(frequencies >= 0)[0]], axis=1) / (
            np.sum(magnitudes, axis=1) + 1e-6)
    res[:, 12] = np.argmax(magnitudes, axis=1)
    res[:, 13], res[:, 14] = hjorth_vectorized(z, diff)
    power_ratio = bin_power_vectorized(z, [0.5, 4, 8, 10, 12, 30], 80)[:]
    res[:, 15:20] = power_ratio[:]
    res[:, 20] = spectral_entropy_vectorized(power_ratio)
    return res


def get_features(x, is_numpy=False):
    x = x if is_numpy else x.cpu().numpy()
    all_features = []
    for ch in range(x.shape[1]):
        z = x[:, ch, :]
        if not np.isfinite(z).all():
            print('INPUT IS NOT FINITE!!')
            exit()
        res = get_features_vectorized(z, True)
        if not np.isfinite(res).all():
            print('OUTPUT IS NOT FINITE!!')
            res[res == -np.inf] = -1e10
            res[res == np.inf] = 1e10
            res[res != res] = 0.0
        all_features.append(res)
    return all_features


def pixel_norm(h):
    mean = torch.mean(h * h, dim=1, keepdim=True)
    dom = torch.rsqrt(mean + 1e-8)
    return h * dom


def simple_argparser(default_params):
    parser = ArgumentParser()
    for k in default_params:
        parser.add_argument('--{}'.format(k), type=partial(generic_arg_parse, hinttype=type(default_params[k])))
    parser.set_defaults(**default_params)
    return get_structured_params(vars(parser.parse_args()))


def enable_benchmark():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True  # for fast training(if network input size is almost constant)


def map_location(storage, location):
    return storage


def load_model(model_path):
    return torch.load(model_path, map_location=map_location)
