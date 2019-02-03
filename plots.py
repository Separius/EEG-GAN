import pickle
import matplotlib
import numpy as np
import pandas as pd
from imageio import imsave

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

max_seq_len = 1024


def load(f_name):
    with open(f_name, 'rb') as f:
        return pickle.load(f)


def get_t(seq_len=1024, frequency=80):
    return np.linspace(0, seq_len / frequency, seq_len)


def running_mean(x, n=6):
    return np.stack([pd.Series(x[:, i]).rolling(window=n).mean().values for i in range(x.shape[1])], axis=1)


def plot_eeg(samples, frequency=80, freq_smooth_factor=6, save_location=None, suptitle=None, mode=None):
    f_factor = 1.0
    if mode == 'slurp':
        no_freq_domain = False
        freq_domain_on_right = True
        slurp_size = samples.shape[0]
        f_factor = 2.0
    elif mode == 'progressive':
        no_freq_domain = False
        freq_domain_on_right = True
        prog_size = len(samples)
        f_factor = 2.0
    elif len(samples) == 6:
        mode = 'knn'
        no_freq_domain = True
        freq_domain_on_right = True
    elif len(samples) == 8:
        mode = 'trunc'
        no_freq_domain = False
        freq_domain_on_right = True
    elif len(samples) == 4:
        mode = 'simple'
        no_freq_domain = False
        freq_domain_on_right = True
    else:
        raise ValueError()
    fig = plt.figure(save_location)
    if freq_domain_on_right:
        fig.set_figwidth(19.2)
        fig.set_figheight(10.8 * f_factor)
    else:
        fig.set_figwidth(19.2 / 2)
        fig.set_figheight(10.8 * 2 * f_factor)
    for index, sample in enumerate(samples):
        sample = sample.transpose()
        n_rows = sample.shape[1]
        seq_len = sample.shape[0]

        def sub_plot(ax, y, x, xlim, crowd, xlabel):
            ax.set_xlim(0, xlim)
            dmin = np.nanmin(y)
            dmax = np.nanmax(y)
            dr = (dmax - dmin) * (1.0 if not crowd else 0.7)  # Crowd them a bit.
            y0 = dmin
            y1 = (n_rows - 1) * dr + dmax
            ax.set_ylim(y0, y1)
            segs = []
            ticklocs = []
            for i in range(n_rows):
                segs.append(np.column_stack((x, y[:, i])))
                ticklocs.append(i * dr)
            offsets = np.zeros((n_rows, 2), dtype=float)
            offsets[:, 1] = ticklocs
            lines = LineCollection(segs, offsets=offsets, transOffset=None)
            ax.add_collection(lines)
            ax.set_yticks(ticklocs)
            ax.set_yticklabels(['CZ', 'O2', 'O1', 'F4', 'F3'])
            ax.set_xlabel(xlabel)

        if mode == 'knn':
            t_ax = fig.add_subplot(3, 2, index + 1)
            if index == 0:
                t_ax.set_title('query')
            else:
                t_ax.set_title(str(index) + ' ' + {1: 'st', 2: 'nd', 3: 'rd', 4: 'th', 5: 'th'}[index] + ' neighbor')
        elif mode == 'trunc':
            t_ax = fig.add_subplot(4, 4, 2 * index + 1)
            if index % 4 == 0:
                t_ax.set_title('t = 0.875')
            elif index % 4 == 1:
                t_ax.set_title('t = 0.9')
            elif index % 4 == 2:
                t_ax.set_title('t = 0.95')
            else:
                t_ax.set_title('t = 1.0')
        elif mode == 'simple':
            t_ax = fig.add_subplot(4, 2, 2 * index + 1)
        elif mode == 'slurp':
            t_ax = fig.add_subplot(slurp_size, 2, 2 * index + 1)
        elif mode == 'progressive':
            t_ax = fig.add_subplot(prog_size, 2, 2 * index + 1)
        this_frequency = frequency * seq_len / max_seq_len
        sub_plot(t_ax, sample, get_t(seq_len, this_frequency), seq_len / this_frequency, True, 'Time (s)')
        if not no_freq_domain:
            sample_f = np.abs(np.fft.rfft(sample, axis=0))
            if freq_smooth_factor is not None and sample_f.shape[0] >= 128:
                sample_f = running_mean(sample_f, freq_smooth_factor)
            if mode == 'trunc':
                f_ax = fig.add_subplot(4, 4, 2 * index + 2)
                if index % 4 == 0:
                    f_ax.set_title('t = 0.875')
                elif index % 4 == 1:
                    f_ax.set_title('t = 0.9')
                elif index % 4 == 2:
                    f_ax.set_title('t = 0.95')
                else:
                    f_ax.set_title('t = 1.0')
            elif mode == 'simple':
                f_ax = fig.add_subplot(4, 2, 2 * index + 2)
            elif mode == 'slurp':
                f_ax = fig.add_subplot(slurp_size, 2, 2 * index + 2)
            elif mode == 'progressive':
                f_ax = fig.add_subplot(prog_size, 2, 2 * index + 2)
            f_ax.grid(True, which='both')
            sub_plot(f_ax, sample_f, np.fft.rfftfreq(seq_len, d=1. / this_frequency), this_frequency / 4, False,
                     'Freq (Hz)')
    plt.tight_layout()
    if suptitle is not None:
        fig.suptitle(suptitle)
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    if save_location is not None:
        imsave(save_location, image)

if __name__ == '__main__':
    base_directory = './015-tuh1_1024_normlatent/network-snapshot-{}-005952.pkl'
    save_location_base = './'

    # input to the plot_eeg is 6 * num_channels * seq_len
    index = 0
    plot_eeg(load(base_directory.format('generator_smooth_5nn_d'))[index],
             save_location=save_location_base + '5nn_d.png',
             suptitle='nearest neighbors based on D(x)')
    plot_eeg(load(base_directory.format('generator_smooth_5nn_freq'))[index],
             save_location=save_location_base + '5nn_freq.png')
    plot_eeg(load(base_directory.format('generator_smooth_5nn_time'))[index],
             save_location=save_location_base + '5nn_time.png')
    plot_eeg(load(base_directory.format('generator_smooth_5nn_freq_ch1'))[index, :, None, :],
             save_location=save_location_base + '5nn_freq_ch1.png')
    plot_eeg(load(base_directory.format('generator_smooth_5nn_time_ch1'))[index, :, None, :],
             save_location=save_location_base + '5nn_time_ch1.png')

    index_one = 0
    index_two = 1
    t_values = [0.875, 0.9, 0.95, 1.0]
    x = np.stack([load(base_directory.format('generator_smooth_t_' + str(t_value)))[index_one] for t_value in t_values],
                 axis=0)
    x = np.concatenate((x, np.stack(
        [load(base_directory.format('generator_smooth_t_' + str(t_value)))[index_two] for t_value in t_values],
        axis=0)),
                       axis=0)
    plot_eeg(x, save_location=save_location_base + 'truncation.png')

    index = 0
    plot_eeg(load(base_directory.format('smooth_generator'))[index:4 + index],
             save_location=save_location_base + 'base.png')
    plot_eeg(load(base_directory.format('generator_smooth_d'))[index:4 + index],
             save_location=save_location_base + 'd_based.png')

    index = 0
    x = load(base_directory.format('generator_smooth_slurp'))[index, ::4]
    plot_eeg(x, save_location=save_location_base + 'slurp.png', mode='slurp')

    index = 0
    x = [d[index] for d in load(base_directory.format('generator_constant'))]
    plot_eeg([x[0], x[5], x[10], x[15], x[20], x[25]], save_location=save_location_base + 'progressive.png',
             mode='progressive')
