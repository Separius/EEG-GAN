import math
import numpy as np
import sklearn as sklearn
from sklearn.decomposition import FastICA
from sklearn.cross_decomposition import CCA


def eye_state(signal, signal_frequency, o1_channel_id=2, o2_channel_id=3, overlap=0.5):
    # signal = (batch_size, num_channels, num_samples)
    delta_t = 2 * signal_frequency
    step = int(math.floor(delta_t * overlap))
    pass


def eye_closure_detection(signal, signal_frequency, o1_channel_id=2, o2_channel_id=3):
    # signal = [n_channels, n_samples]
    # TODO batchify this
    overlap = 0.5
    delta_t = 2 * signal_frequency
    step = np.floor(delta_t * overlap)
    step = step.astype('int32')
    min_frequency = 3
    min_freq_alpha = 7
    max_freq_alpha = 13
    num_of_channels = signal.shape[0]
    alpha_mode = False
    delta_f_cca = np.floor((min_freq_alpha + max_freq_alpha) / 2 - min_frequency - 1) / 3
    freqs = np.array([min_frequency + 1, min_frequency + 1 + delta_f_cca, min_frequency + 1 + 2 * delta_f_cca,
                      (min_freq_alpha + max_freq_alpha) / 2, max_freq_alpha + delta_f_cca])
    signal = sklearn.preprocessing.normalize(signal, norm='l2', axis=1, copy=True, return_norm=False)
    res_size = 0
    for l in range(0, signal.shape[1] - delta_t, step):
        res_size += 1
    # ICA
    transformer = FastICA(random_state=0)
    C = transformer.fit_transform(np.transpose(signal))
    C = np.transpose(C)
    corr = np.absolute(np.corrcoef(np.concatenate((signal, C), axis=0)))
    corr = corr[:num_of_channels, num_of_channels:]
    corr = (corr[o1_channel_id, :] + corr[o2_channel_id, :]) / 2
    o_component = np.argmax(corr)

    # CCA

    t = np.arange(0, delta_t) / signal_frequency
    sin1 = [np.sin(2 * np.pi * freqs[0] * t)]
    sin2 = [np.sin(2 * np.pi * freqs[1] * t)]
    sin3 = [np.sin(2 * np.pi * freqs[2] * t)]
    sin4 = [np.sin(2 * np.pi * freqs[3] * t)]
    sin5 = [np.sin(2 * np.pi * freqs[4] * t)]

    SIN = np.concatenate((sin1, sin2, sin3, sin4, sin5), axis=0)

    alpha_label = np.zeros((res_size,))
    alpha_label_adaptive = np.zeros((res_size,))
    alpha_label_3level = np.zeros((res_size,))

    # whole signal main frquency
    # TODO batchify these two
    FFT = np.fft.fft(signal[o1_channel_id, :])
    f = np.absolute(signal_frequency * np.fft.fftfreq(len(FFT)))
    FFT = FFT[np.argwhere(f >= min_frequency)[0][0]: len(FFT) / 2]
    f = f[np.argwhere(f >= min_frequency)[0][0]: len(FFT) / 2]
    idx = np.argmax(np.abs(FFT))
    main_freq_o1 = f[idx]

    FFT = np.fft.fft(signal[o2_channel_id, :])
    f = np.absolute(signal_frequency * np.fft.fftfreq(len(FFT)))
    FFT = FFT[np.argwhere(f >= min_frequency)[0][0]: len(FFT) / 2]
    f = f[np.argwhere(f >= min_frequency)[0][0]: len(FFT) / 2]
    idx = np.argmax(np.abs(FFT))
    main_freq_o2 = f[idx]

    if np.mean([main_freq_o1, main_freq_o2]) >= min_freq_alpha and np.mean(
            [main_freq_o1, main_freq_o2]) <= max_freq_alpha:
        alpha_mode = True

    # TODO batchify this
    for kk, l in enumerate(range(0, signal.shape[1] - delta_t, step)):
        # ICA temporal
        FFT = np.fft.fft(C[o_component, l:l + delta_t])
        f = np.absolute(signal_frequency * np.fft.fftfreq(len(FFT)))
        FFT = FFT[np.argwhere(f >= min_frequency)[0][0]: len(FFT) / 2]
        f = f[np.argwhere(f >= min_frequency)[0][0]: len(FFT) / 2]
        idx = np.argmax(np.abs(FFT))
        main_freq_ICA = f[idx]

        # CCA temporal
        cca = CCA(n_components=1)
        signal_c, SIN_c = cca.fit_transform(np.transpose(signal[[o1_channel_id, o2_channel_id], l:l + delta_t]),
                                            np.transpose(SIN))
        corr = np.absolute(np.corrcoef(np.concatenate((np.transpose(SIN_c), SIN), axis=0)))
        corr = corr[0:SIN_c.shape[1], SIN_c.shape[1]:]
        main_freq_CCA = freqs[np.argmax(corr)]
        main_freq_avg = (main_freq_CCA + main_freq_ICA) / 2
        repeated_cond = ((max_freq_alpha >= main_freq_ICA >= min_freq_alpha) or (
                max_freq_alpha >= main_freq_CCA >= min_freq_alpha)) and (
                                max_freq_alpha >= main_freq_avg >= min_freq_alpha)

        if alpha_mode:
            if max_freq_alpha >= main_freq_ICA >= min_freq_alpha:
                alpha_label_adaptive[kk] = 1
        else:
            if repeated_cond:
                alpha_label_adaptive[kk] = 1

        if repeated_cond:
            alpha_label[kk] = 1

        if max_freq_alpha >= main_freq_ICA >= min_freq_alpha:
            alpha_label_3level[kk] = 0.5
        else:
            if repeated_cond:
                alpha_label_3level[kk] = 1
    return alpha_label, alpha_label_adaptive, alpha_label_3level
