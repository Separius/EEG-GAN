from torch.autograd import Variable
from utils import cudize, random_latents, generate_samples, simple_argparser
import torch
import pickle
import glob
from plugins import OutputGenerator
from scipy import misc
import os

default_params = {
    'generator_path': '',
    'num_samples': 128,
    'is_dir': False,
    'num_pics': 0,
    'frequency': 80
}


def output_samples(generator_path, num_samples):
    G = torch.load(generator_path, map_location=lambda storage, location: storage)
    G = cudize(G)
    gen_input = cudize(Variable(random_latents(num_samples, G.latent_size)))
    output = generate_samples(G, gen_input)
    return output


def save_pics(xx, generator):
    if params['num_pics'] != 0:
        images = OutputGenerator.get_images(xx.shape[2], params['frequency'], 0, xx[:params['num_pics'], ...])
        for i, image in enumerate(images):
            misc.imsave(generator.replace('.dat', '_{}.png').format(i), image)


if __name__ == '__main__':
    params = simple_argparser(default_params)
    if params['is_dir']:
        params['generator_path'] = os.path.join(params['generator_path'], 'network-snapshot-generator-*.dat')
        for generator in glob.glob(params['generator_path']):
            xx = output_samples(generator, params['num_samples'])
            pickle.dump(xx, open(generator.replace('.dat', '.pkl'), 'wb'))
            save_pics(xx, generator)
    else:
        xx = output_samples(params['generator_path'], params['num_samples'])
        pickle.dump(xx, open(params['generator_path'].replace('.dat', '.pkl'), 'wb'))
        save_pics(xx, params['generator_path'])
