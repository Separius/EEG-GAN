import os
import torch
import pickle
import numpy as np
from tqdm import trange, tqdm
from dataset import EEGDataset
from plugins import DepthManager
from network import Generator, Discriminator
from utils import cudize, random_latents, load_model, parse_config

default_params = {
    'config_file': None,
    'num_data_workers': 1,
    'random_seed': 1373,
    'cuda_device': 0,
    'resume_network': '',
    'num_samples': 128,
    'frequency': 80,
}


def get_random_latents(bs):
    y = cudize(dataset.generate_class_condition(bs))
    z = cudize(random_latents(bs, generator.latent_size, generator.z_distribution))
    if y is None:
        return {'z': z}
    y[bs // 2:] = y[:bs // 2]
    return {'z': z, 'y': y}


if __name__ == '__main__':
    params = parse_config(default_params, [EEGDataset, Generator, Discriminator, DepthManager], False)
    dataset_params = params['EEGDataset']
    dataset, val_dataset = EEGDataset.from_config(**dataset_params)
    num_classes = 0 if dataset.y is None or dataset.no_condition else dataset.y.shape[1]
    shared_model_params = dict(dataset_shape=dataset.shape, initial_size=dataset.model_dataset_depth_offset,
                               fmap_base=params['fmap_base'], fmap_max=params['fmap_max'], init=params['init'],
                               fmap_min=params['fmap_min'], kernel_size=params['kernel_size'],
                               residual=params['residual'], equalized=params['equalized'],
                               sagan_non_local=params['sagan_non_local'],
                               average_conditions=params['average_conditions'],
                               factorized_attention=params['use_factorized_attention'],
                               self_attention_layers=params['self_attention_layers'], act_alpha=params['act_alpha'],
                               num_classes=num_classes, progression_scale=dataset.progression_scale)
    generator = Generator(**shared_model_params, z_distribution=params['z_distribution'], **params['Generator'])
    generator_smooth = Generator(**shared_model_params, z_distribution=params['z_distribution'], **params['Generator'])
    discriminator = Discriminator(**shared_model_params, **params['Discriminator'])
    dest = os.path.join(params['result_dir'], params['resume_network'])
    generator_state, _, g_cur_img = load_model(dest.format('generator'), True)
    discriminator_state, _, d_cur_img = load_model(dest.format('discriminator'), True)
    assert g_cur_img == d_cur_img
    generator.load_state_dict(generator_state)
    discriminator.load_state_dict(discriminator_state)
    generator_smooth.load_state_dict(torch.load(dest.format('smooth_generator'), map_location='cpu')['model'])
    dm = DepthManager(None, None, generator.max_depth, None, len(params['self_attention_layers']) != 0, None, None,
                      **params['DepthManager'])
    depth, alpha = dm.calc_progress(g_cur_img)
    dataset.alpha = val_dataset.alpha = alpha
    generator.alpha = discriminator.alpha = generator_smooth.alpha = alpha
    dataset.model_depth = val_dataset.model_depth = depth
    generator.depth = discriminator.depth = generator_smooth.depth = depth
    generator = cudize(generator)
    discriminator = cudize(discriminator)
    generator_smooth = cudize(generator_smooth)

    with torch.no_grad():
        g_loss_mean = 0.0
        burn_out_size = 50
        for i in trange(burn_out_size):
            z = get_random_latents(params['num_samples'])
            x, _ = generator(z)
            g_loss = -discriminator(x, z.get('y', None))[0]
            g_loss_mean += g_loss.mean().item()
        g_loss_mean /= burn_out_size
        print('g_loss_mean', g_loss_mean)

        accepted_results = []
        l = [None] + [i for i in range(generator.depth - 1)]
        for i in tqdm(l):
            total_accepted = 0
            g_loss_mean_fake = 0.0
            num_batches = 50
            for j in trange(num_batches):
                z = get_random_latents(params['num_samples'])
                x, _ = generator(z, merge_layer=i)
                y = z.get('y', None)
                if y is not None:
                    y = y[:y.size(0) // 2]
                g_loss = -discriminator(x, y)[0]
                g_loss_mean_fake += g_loss.mean().item()
                fake_accepted = x[g_loss < g_loss_mean].data.cpu().numpy()
                if i is not None:
                    accepted_results.append(fake_accepted)
                total_accepted += fake_accepted.shape[0]
            print(i, total_accepted, g_loss_mean_fake / num_batches)
    accepted_results = np.concatenate(accepted_results, axis=0)
    dest = os.path.join(params['result_dir'], params['resume_network'])
    print(accepted_results.shape)
    pickle.dump(accepted_results, open(dest.format('extended').replace('.dat', '.pkl'), 'wb'))
