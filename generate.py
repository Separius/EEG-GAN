import os
import glob
import math
import pickle
import numpy as np
from scipy import misc
from tqdm import trange, tqdm
from torch.autograd import Variable
from plugins import OutputGenerator
from utils import cudize, random_latents, generate_samples, simple_argparser, load_model

default_params = {
    'generator_path': '',
    'num_samples': 1024,
    'num_pics': 0,
    'frequency': 80,
    'max_batch_size': 128
}


def output_samples(generator_path, num_samples):
    G = load_model(generator_path)
    G = cudize(G)
    if num_samples < params['max_batch_size']:
        params['max_batch_size'] = num_samples
    outputs = []
    for i in trange(int(math.ceil(num_samples / params['max_batch_size']))):
        z = random_latents(params['max_batch_size'], G.latent_size, 8 if G.is_extended else 1)
        if not isinstance(z, (tuple, list)):
            z = (z, )
        gen_input = (cudize(Variable(x)) for x in z)
        outputs.append(generate_samples(G, gen_input))
    return np.concatenate(outputs, axis=0)


def save_pics(xx, generator):
    if params['num_pics'] != 0:
        images = OutputGenerator.get_images(xx.shape[2], params['frequency'], 0, xx[:params['num_pics'], ...])
        for i in trange(len(images)):
            misc.imsave(generator.replace('.dat', '_{}.png').format(i), images[i])


if __name__ == '__main__':
    params = simple_argparser(default_params)
    if os.path.isdir(params['generator_path']):
        params['generator_path'] = os.path.join(params['generator_path'], '*-network-snapshot-generator-*.dat')
        all_generators = glob.glob(params['generator_path'])
        for generator in tqdm(all_generators):
            xx = output_samples(generator, params['num_samples'])
            pickle.dump(xx, open(generator.replace('.dat', '.pkl'), 'wb'))
            save_pics(xx, generator)
    else:
        xx = output_samples(params['generator_path'], params['num_samples'])
        pickle.dump(xx, open(params['generator_path'].replace('.dat', '.pkl'), 'wb'))
        save_pics(xx, params['generator_path'])
