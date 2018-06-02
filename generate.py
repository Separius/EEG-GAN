from torch.autograd import Variable
from utils import cudize, random_latents, generic_arg_parse, generate_samples, get_structured_params
import torch
from argparse import ArgumentParser
from functools import partial
import pickle
import glob

default_params = {
    'generator_path': '',
    'num_samples': 128,
    'output_path': '',
    'is_dir': False
}


def output_samples(generator_path, num_samples):
    G = torch.load(generator_path, map_location=lambda storage, location: storage)
    G = cudize(G)
    latent_size = getattr(G, 'latent_size', 512)
    gen_input = cudize(Variable(random_latents(num_samples, latent_size)))
    output = generate_samples(G, gen_input)
    return output


if __name__ == '__main__':
    parser = ArgumentParser()
    for k in default_params:
        parser.add_argument('--{}'.format(k), type=partial(generic_arg_parse, hinttype=type(default_params[k])))
    parser.set_defaults(**default_params)
    params = get_structured_params(vars(parser.parse_args()))
    if params['is_dir']:
        params['generator_path'] = params['generator_path'] + '*'
        for generator in glob.glob(params['generator_path']):
            xx = output_samples(generator, params['num_samples'])
            pickle.dump(xx, open(generator.replace('.dat', '.pkl'), 'wb'))
    else:
        xx = output_samples(params['generator_path'], params['num_samples'])
        pickle.dump(xx, open(params['generator_path'], 'wb'))
