from torch.autograd import Variable
from utils import cudize, random_latents, generate_samples, simple_argparser, load_model
import pickle
import glob
from plugins import OutputGenerator
from scipy import misc
import os
from tqdm import trange

default_params = {
    'generator_path': '',
    'num_samples': 128,
    'num_pics': 8,
    'frequency': 80
}


def output_samples(generator_path, num_samples):
    G = load_model(generator_path)
    G = cudize(G)
    gen_input = cudize(Variable(random_latents(num_samples, G.latent_size)))  # TODO this is wrong
    output = generate_samples(G, gen_input)
    return output


def save_pics(xx, generator):
    if params['num_pics'] != 0:
        images = OutputGenerator.get_images(xx.shape[2], params['frequency'], 0, xx[:params['num_pics'], ...])
        for i in trange(len(images)):
            misc.imsave(generator.replace('.dat', '_{}.png').format(i), images[i])


if __name__ == '__main__':
    params = simple_argparser(default_params)
    if os.path.isdir(params['generator_path']):
        params['generator_path'] = os.path.join(params['generator_path'], 'network-snapshot-generator-*.dat')
        all_generators = glob.glob(params['generator_path'])
        for i in trange(len(all_generators)):
            generator = all_generators[i]
            xx = output_samples(generator, params['num_samples'])
            pickle.dump(xx, open(generator.replace('.dat', '.pkl'), 'wb'))
            save_pics(xx, generator)
    else:
        xx = output_samples(params['generator_path'], params['num_samples'])
        pickle.dump(xx, open(params['generator_path'].replace('.dat', '.pkl'), 'wb'))
        save_pics(xx, params['generator_path'])
