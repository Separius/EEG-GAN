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


def plot_eeg(samples, frequency=80, freq_smooth_factor=6, save_location=None, suptitle=None):
    if len(samples) == 6:
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
    elif len(samples) == 5:
        mode = 'slurp'
        no_freq_domain = False
        freq_domain_on_right = True
    elif len(samples) == 3:
        mode = 'progressive'
        no_freq_domain = False
        freq_domain_on_right = True
    else:
        raise ValueError()
    fig = plt.figure(save_location)
    if freq_domain_on_right:
        fig.set_figwidth(19.2)
        fig.set_figheight(10.8)
    else:
        fig.set_figwidth(19.2 / 2)
        fig.set_figheight(10.8 * 2)
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
        elif mode == 'trunc':
            t_ax = fig.add_subplot(4, 4, 2 * index + 1)
        elif mode == 'simple':
            t_ax = fig.add_subplot(4, 2, 2 * index + 1)
        elif mode == 'slurp':
            t_ax = fig.add_subplot(5, 2, 2 * index + 1)
        elif mode == 'progressive':
            t_ax = fig.add_subplot(3, 2, 2 * index + 1)
        else:
            t_ax = fig.gca() if no_freq_domain else (
                fig.add_subplot(1, 2, 1) if freq_domain_on_right else fig.add_subplot(2, 1, 1))
        sub_plot(t_ax, sample, get_t(seq_len, frequency), seq_len / frequency, True, 'Time (s)')
        if not no_freq_domain:
            sample_f = np.abs(np.fft.rfft(sample, axis=0))
            if freq_smooth_factor is not None and sample_f.shape[0] >= 32:
                sample_f = running_mean(sample_f, freq_smooth_factor)
            if mode == 'trunc':
                f_ax = fig.add_subplot(4, 4, 2 * index + 2)
            elif mode == 'simple':
                f_ax = fig.add_subplot(4, 2, 2 * index + 2)
            elif mode == 'slurp':
                f_ax = fig.add_subplot(5, 2, 2 * index + 2)
            elif mode == 'progressive':
                f_ax = fig.add_subplot(3, 2, 2 * index + 2)
            else:
                f_ax = fig.add_subplot(1, 2, 2) if freq_domain_on_right else fig.add_subplot(2, 1, 2)
            this_frequency = frequency * seq_len / max_seq_len
            sub_plot(f_ax, sample_f, np.fft.rfftfreq(seq_len, d=1. / this_frequency), this_frequency / 2, False,
                     'Freq (Hz)')
    plt.tight_layout()
    if suptitle is not None:
        fig.suptitle(suptitle)
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    if save_location is not None:
        imsave(save_location, image)


base_directory = './results/015-tuh1_1024_normlatent/network-snapshot-{}-005952.pkl'
index = 0
plot_eeg(load(base_directory.format('generator_smooth_5nn_d'))[index], save_location='/home/sepehr/Desktop/5nn_d.png')
plot_eeg(load(base_directory.format('generator_smooth_5nn_freq'))[index],
         save_location='/home/sepehr/Desktop/5nn_freq.png')
plot_eeg(load(base_directory.format('generator_smooth_5nn_time'))[index],
         save_location='/home/sepehr/Desktop/5nn_time.png')
plot_eeg(load(base_directory.format('generator_smooth_5nn_freq_ch1'))[index, :, None, :],
         save_location='/home/sepehr/Desktop/5nn_freq_ch1.png')
plot_eeg(load(base_directory.format('generator_smooth_5nn_time_ch1'))[index, :, None, :],
         save_location='/home/sepehr/Desktop/5nn_time_ch1.png')

index_one = 0
index_two = 1
t_values = [0.875, 0.9, 0.95, 1.0]
x = np.stack([load(base_directory.format('generator_smooth_t_' + str(t_value)))[index_one] for t_value in t_values],
             axis=0)
x = np.concatenate((x, np.stack(
    [load(base_directory.format('generator_smooth_t_' + str(t_value)))[index_two] for t_value in t_values], axis=0)),
                   axis=0)
plot_eeg(x, save_location='/home/sepehr/Desktop/truncation.png')

index = 0
plot_eeg(load(base_directory.format('smooth_generator'))[index:4 + index],
         save_location='/home/sepehr/Desktop/base.png')
plot_eeg(load(base_directory.format('generator_smooth_d'))[index:4 + index],
         save_location='/home/sepehr/Desktop/d_based.png')

index = 0
plot_eeg(load(base_directory.format('generator_smooth_slurp'))[index, ::7],
         save_location='/home/sepehr/Desktop/slurp.png')

index = 0
x = [d[index] for d in load(base_directory.format('generator_constant'))]
plot_eeg([x[0], x[10], x[20]], save_location='/home/sepehr/Desktop/progressive.png')
