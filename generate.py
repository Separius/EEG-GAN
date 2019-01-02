import os
import math
import glob
import torch
import faiss
import pickle
import imageio
import numpy as np
from scipy import misc
from tqdm import trange, tqdm
from dataset import EEGDataset
from network import Generator, Discriminator
from plugins import OutputGenerator, DepthManager
from utils import cudize, random_latents, generate_samples, load_model, parse_config
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

default_params = {
    'config_file': None,
    'num_data_workers': 1,
    'random_seed': 1373,
    'cuda_device': 0,
    'resume_network': '',
    'num_samples': 128,
    'num_pics': 4,
    'frequency': 80,
    'max_batch_size': 128,
    'g_loss_threshold': -1.5,
}


# TODO Extract saliency maps of discriminator
# TODO visualize attention map of G and D
# TODO condition interpolation * Z(*chunks) interpolation
# TODO FID-IS AuC for truncation trick

def get_random_latents(bs):
    y = dataset.generate_class_condition(bs)
    z = random_latents(bs, generator.latent_size, generator.z_distribution)
    if y is None:
        return {'z': z}
    return {'z': z, 'y': y}


def output_samples():
    if params['num_samples'] < params['max_batch_size']:
        params['max_batch_size'] = params['num_samples']
    outputs = []
    outputs_smooth = []
    for i in trange(int(math.ceil(params['num_samples'] / params['max_batch_size']))):
        z = cudize(get_random_latents(params['max_batch_size']))
        outputs.append(generate_samples(generator, z))
        outputs_smooth.append(generate_samples(generator_smooth, z))
    return np.concatenate(outputs, axis=0), np.concatenate(outputs_smooth, axis=0)


def output_samples_based_on_d(given_generator):
    outputs = []
    total_len = 0
    tqdm_counter = tqdm(total=params['num_samples'])
    while total_len < params['num_samples']:
        z = cudize(get_random_latents(params['max_batch_size']))
        fake = given_generator(z)[0]
        g_loss = -discriminator(fake, z.get('y', None))[0]
        fake = fake[g_loss < params['g_loss_threshold']].data.cpu().numpy()
        outputs.append(fake)
        total_len += len(fake)
        tqdm_counter.update(len(fake))
    tqdm_counter.close()
    return np.concatenate(outputs, axis=0)[:params['num_samples']]


def output_samples_with_truncation(given_generator, thresh=0.95):
    outputs = []
    total_len = 0
    tqdm_counter = tqdm(total=params['num_samples'])
    while total_len < params['num_samples']:
        z = cudize(get_random_latents(params['max_batch_size']))
        mask = (z['z'].norm(p=2, dim=1) / math.sqrt(given_generator.latent_size)) <= thresh
        z = {k: v[mask] for k, v in z.items()}
        if len(z['z']) == 0:
            continue
        fake = generate_samples(given_generator, z)
        outputs.append(fake)
        total_len += len(fake)
        tqdm_counter.update(len(fake))
    tqdm_counter.close()
    return np.concatenate(outputs, axis=0)[:params['num_samples']]


def save_pics(samples, loc):
    if params['num_pics'] != 0:
        images = OutputGenerator.get_images(samples.shape[2],
                                            params['frequency'] / (dataset.shape[2] / samples.shape[2]), 0,
                                            samples[:params['num_pics']])
        for i in trange(len(images)):
            misc.imsave(loc.replace('.dat', '_{}.png').format(i), images[i])


def slerp(val, low, high):
    omega = np.arccos(np.clip(np.dot(low / np.linalg.norm(low), high / np.linalg.norm(high)), -1, 1))
    so = np.sin(omega)
    if so == 0:
        return np.outer(1.0 - val, low) + np.outer(val, high)
    return np.outer(np.sin((1.0 - val) * omega) / so, low) + np.outer(np.sin(val * omega) / so, high)


def z1_to_z2(given_generator, dest, num_pics=16, get_out=False):
    z = get_random_latents(2)
    t = np.arange(num_pics) / num_pics
    z['z'] = torch.from_numpy(slerp(t, z['z'][0], z['z'][1]).astype(np.float32))
    if 'y' in z:
        pass  # TODO
    z = cudize(z)
    out = generate_samples(given_generator, z)
    if get_out:
        return out
    frequency = params['frequency'] * out.shape[2] / dataset.shape[2]
    images = OutputGenerator.get_images(out.shape[2], frequency, 0, out, trange)
    imageio.mimsave(dest, images, fps=2)


def plot_knn(data):
    seq_len, frequency = data.shape[-1], params['frequency'] / (dataset.shape[2] / data.shape[-1])
    num_channels = data.shape[2]
    # data is 8(num_signals) * 6(k+1) * num_channels * length
    t = np.linspace(0, seq_len / frequency, seq_len)
    images = []
    for index in range(len(data)):
        fig = plt.figure(figsize=(40, 40))
        outer = gridspec.GridSpec(3, 2, wspace=0.2, hspace=0.2)
        for i in range(6):
            inner = gridspec.GridSpecFromSubplotSpec(num_channels, 1, subplot_spec=outer[i], wspace=0.1, hspace=0.1)
            for j in range(num_channels):
                ax = plt.Subplot(fig, inner[j])
                ax.plot(t, data[index, i, j], color=(0.8, 0, 0, 0.5))
                ax.set_ylim([-1.1, 1.1])
                fig.add_subplot(ax)
        fig.canvas.draw()
        image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images.append(image)
        plt.close(fig)
    return images


def gzip():
    return zip([generator, generator_smooth], ['generator', 'generator_smooth'])


def plot_over_time(datas, depths):
    images = []
    for fake, depth in tqdm(zip(datas, depths)):
        frequency = params['frequency'] * fake.shape[2] / dataset.shape[2]
        seq_len = fake.shape[2]
        num_channels = fake.shape[1]
        t = np.linspace(0, seq_len / frequency, seq_len)
        fig, (axs) = plt.subplots(num_channels, 4)
        fig.set_figheight(40)
        fig.set_figwidth(40)
        for ch in range(num_channels):
            for index in range(4):
                axs[ch][index].plot(t, fake[index, ch], color=(0.8, 0, 0, 0.5), label='time domain')
                axs[ch][index].set_ylim([-1.1, 1.1])
        fig.suptitle('depth: {}'.format(depth))
        fig.canvas.draw()
        image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images.append(image)
        plt.close(fig)
    return images


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
        # z1 to z2: gif
        for i in trange(2):
            for g, name in gzip:
                z1_to_z2(g, dest.format(name).replace('.dat', '_{}.gif'.format(i)))

        # z1 to z2: one plot
        for g, name in gzip():
            data = np.stack([z1_to_z2(g, '', num_pics=6, get_out=True) for i in range(8)], axis=0)
            images = plot_knn(data)
            for i, image in enumerate(images):
                misc.imsave(dest.format(name + '_slurp').replace('.dat', '_{}.png'.format(i)), image)

        # truncation trick
        for thresh in tqdm([1.0, 0.95, 0.9, 0.875]):
            for g, name in gzip():
                samples_gen = output_samples_with_truncation(g, thresh)
                this_dest = dest.format(name + '_t_' + str(thresh))
                pickle.dump(samples_gen, open(this_dest.replace('.dat', '.pkl'), 'wb'))
                save_pics(samples_gen, this_dest)

        # basic generation with same z
        samples_gen, samples_smooth = output_samples()
        pickle.dump(samples_gen, open(dest.format('generator').replace('.dat', '.pkl'), 'wb'))
        pickle.dump(samples_smooth, open(dest.format('smooth_generator').replace('.dat', '.pkl'), 'wb'))
        save_pics(samples_gen, dest.format('generator'))
        save_pics(samples_smooth, dest.format('smooth_generator'))

        # basic generation based on D value
        if params['loss_type'] == 'hinge' or params['loss_type'].startswith('wgan'):
            for g, name in gzip():
                samples_gen = output_samples_based_on_d(g)
                pickle.dump(samples_gen, open(dest.format(name + '_d').replace('.dat', '.pkl'), 'wb'))
                save_pics(samples_gen, dest.format(name + '_d'))

    d = dataset[0]['x'].shape[0] * dataset[0]['x'].shape[1]
    index = faiss.IndexFlatL2(d)
    for i in trange(len(dataset)):
        index.add(dataset[i]['x'].view(1, -1).numpy())
    num_samples = 8

    with torch.no_grad():
        # knn in time domain
        for g, name in gzip():
            fake = generate_samples(g, cudize(get_random_latents(num_samples)))
            D, I = index.search(fake.reshape((fake.shape[0], -1)), 5)
            images = plot_knn(np.stack(
                [np.stack([fake[i]] + [dataset[j]['x'].numpy() for j in I[i]], axis=0) for i in range(len(fake))],
                axis=0))
            for i, image in enumerate(images):
                misc.imsave(dest.format(name + '_5nn_time').replace('.dat', '{}.png'.format(i)), image)


    def freq_features(data):
        return np.concatenate([np.abs(np.fft.rfft(data[j])) for j in range(len(data))]).reshape((1, -1)).astype(
            np.float32)


    index = None
    for i in trange(len(dataset)):
        data = freq_features(dataset[i]['x'].numpy())
        if i == 0:
            index = faiss.IndexFlatL2(data.shape[1])
        index.add(data)

    with torch.no_grad():
        # knn in freq domain
        for g, name in gzip():
            fake = generate_samples(g, cudize(get_random_latents(num_samples)))
            D, I = index.search(np.concatenate([freq_features(fake[i]) for i in range(len(fake))]), 5)
            images = plot_knn(np.stack(
                [np.stack([fake[i]] + [dataset[j]['x'].numpy() for j in I[i]], axis=0) for i in range(len(fake))],
                axis=0))
            for i, image in enumerate(images):
                misc.imsave(dest.format(name + '_5nn_freq').replace('.dat', '{}.png'.format(i)), image)

    const_z = cudize(get_random_latents(4))
    fakes = []
    reals = []
    depths = []
    with torch.no_grad():
        d = dest.format('generator')
        for g_address in sorted(glob.glob(d[:d.rfind('-')] + '-*.dat')):
            generator_state, _, g_cur_img = load_model(g_address, True)
            generator.load_state_dict(generator_state)
            depth, alpha = dm.calc_progress(g_cur_img)
            dataset.alpha = val_dataset.alpha = alpha
            generator.alpha = discriminator.alpha = generator_smooth.alpha = alpha
            generator.alpha = alpha
            dataset.model_depth = val_dataset.model_depth = depth
            generator.depth = discriminator.depth = generator_smooth.depth = depth
            generator.depth = depth
            generator = cudize(generator)
            fakes.append(generate_samples(generator, const_z))
            reals.append(np.stack([dataset[i]['x'].numpy() for i in range(4)], axis=0))
            depths.append(depth + alpha)
    fakes_over_time = plot_over_time(fakes, depths)
    imageio.mimsave(dest.format('generator_constant').replace('.dat', '.gif'), fakes_over_time, fps=1)
    reals_over_time = plot_over_time(reals, depths)
    imageio.mimsave(dest.format('real_constant').replace('.dat', '.gif'), reals_over_time, fps=1)
