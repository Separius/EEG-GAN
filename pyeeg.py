import numpy as np


def bin_power_vectorized(X, Band, Fs):
    C = np.fft.fft(X, axis=1)
    C = np.abs(C)
    Power = np.zeros((X.shape[0], len(Band) - 1))
    for Freq_Index in range(len(Band) - 1):
        Freq = float(Band[Freq_Index])
        Next_Freq = float(Band[Freq_Index + 1])
        Power[:, Freq_Index] = C[:, int(Freq / Fs * len(X)):int(Next_Freq / Fs * len(X))].sum(axis=1)
    Power_Ratio = Power / (Power.sum(axis=1, keepdims=True) + 1e-6)
    return Power_Ratio


def hjorth_vectorized(X, D):
    n = X.shape[1] - 1
    M2 = np.square(D).sum(axis=1) / n + 1e-6
    TP = np.square(X).sum(axis=1) + 1e-6
    M4 = np.square(np.diff(D)).sum(axis=1) / n
    return np.sqrt(M2 / TP), np.sqrt(M4 * TP / np.square(M2))


def spectral_entropy_vectorized(Power_Ratio):
    return -np.sum(Power_Ratio * np.log(Power_Ratio + 1e-6), axis=1) / np.log(Power_Ratio.shape[1])
