from torch.autograd import Variable
from utils import *
from argparse import ArgumentParser
from functools import partial

default_params = {
    'generator_path': '',
    'num_samples': 6,
    'description': 'unknown',
}


def output_samples(generator_path, num_samples, description):
    G = torch.load(generator_path, map_location=lambda storage, location: storage)
    G = cudize(G)
    latent_size = getattr(G, 'latent_size', 512)  # yup I just want to use old checkpoints
    print('Sampling noise...')
    gen_input = cudize(Variable(random_latents(num_samples, latent_size)))
    print('Generating...')
    output = generate_samples(G, gen_input)
    print('Done.')
    return output


if __name__ == '__main__':
    parser = ArgumentParser()
    for k in default_params:
        parser.add_argument('--{}'.format(k), type=partial(generic_arg_parse, hinttype=type(default_params[k])))
    parser.set_defaults(**default_params)
    params = get_structured_params(vars(parser.parse_args()))
    xx = output_samples(params['generator_path'], params['num_samples'], params['description'])
    import pickle
    pickle.dump(xx, open('data.gen', 'wb'))
