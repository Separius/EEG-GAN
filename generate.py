import os
import glob
import math
import pickle
import numpy as np
from scipy import misc
from tqdm import trange, tqdm
from dataset import EEGDataset
from torch.autograd import Variable
from plugins import OutputGenerator
from network import Generator, Discriminator
from utils import cudize, random_latents, generate_samples, load_model, parse_config

default_params = {
    'config_file': None,
    'cpu_deterministic': True,
    'num_data_workers': 1,
    'random_seed': 1373,
    'cuda_device': 0,
    'generator_path': '',
    'num_samples': 1024,
    'num_pics': 0,
    'frequency': 80,
    'max_batch_size': 128
}


# TODO sampling based on D value
# TODO sampling based on rejection sampling: https://github.com/shinseung428/DRS_Tensorflow
# TODO sampling with truncation trick: truncation_tirck (of norm(z) > threshold, resample)
# TODO visualize attention map of G and D
# TODO picture: top_generated, bottom 5 nearest neighbors in feature_space of D / feature_space of Inception / MSE(time) / MSE(freq)
# TODO gif(z1 -> z2)
# TODO gif(const_z over epochs)
# TODO aggregation graphs (like the eeg paper) in freq domain
# TODO condition interpolation * Z(*chunks) interpolation
# TODO FID-IS AuC for truncation trick


def output_samples(generator_path, num_samples):
    G = cudize(load_model(generator_path))  # TODO change to first instantiate a G
    if num_samples < params['max_batch_size']:
        params['max_batch_size'] = num_samples
    outputs = []
    # TODO class condition by instantiating a DataSet
    for i in trange(int(math.ceil(num_samples / params['max_batch_size']))):
        outputs.append(generate_samples(G, cudize(
            Variable(random_latents(params['max_batch_size'], G.latent_size, G.z_distribution)))))
    return np.concatenate(outputs, axis=0)


def save_pics(xx, generator):
    if params['num_pics'] != 0:
        images = OutputGenerator.get_images(xx.shape[2], params['frequency'], 0, xx[:params['num_pics']])
        for i in trange(len(images)):
            misc.imsave(generator.replace('.dat', '_{}.png').format(i), images[i])


if __name__ == '__main__':
    params = parse_config(default_params, [])
    num_classes = 0 if dataset.class_options is None else sum(dataset.class_options)
    shared_model_params = dict(dataset_shape=dataset.shape, initial_size=dataset.model_dataset_depth_offset,
                               fmap_base=params['fmap_base'], fmap_max=params['fmap_max'], init=params['init'],
                               fmap_min=params['fmap_min'], kernel_size=params['kernel_size'],
                               residual=params['residual'], equalized=params['equalized'],
                               sagan_non_local=params['sagan_non_local'],
                               average_conditions=params['average_conditions'],
                               factorized_attention=params['use_factorized_attention'],
                               self_attention_layers=params['self_attention_layers'], act_alpha=params['act_alpha'],
                               num_classes=num_classes, progression_scale=dataset.progression_scale)
    generator = Generator
    if os.path.isdir(params['generator_path']):
        params['generator_path'] = os.path.join(params['generator_path'], '*-network-snapshot-generator-*.dat')
        all_generators = glob.glob(params['generator_path'])
    else:
        all_generators = [params['generator_path']]
    for generator in tqdm(all_generators):
        xx = output_samples(generator, params['num_samples'])
        pickle.dump(xx, open(generator.replace('.dat', '.pkl'), 'wb'))
        save_pics(xx, generator)
