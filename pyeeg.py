import numpy as np


def embed_seq(time_series, tau, embedding_dimension):
    if not type(time_series) == np.ndarray:
        typed_time_series = np.asarray(time_series)
    else:
        typed_time_series = time_series
    shape = (typed_time_series.size - tau * (embedding_dimension - 1), embedding_dimension)
    strides = (typed_time_series.itemsize, tau * typed_time_series.itemsize)
    return np.lib.stride_tricks.as_strided(typed_time_series, shape=shape, strides=strides)


def dfa(X):
    """Compute Detrended Fluctuation Analysis from a time series X"""
    X = np.array(X)
    Ave = np.mean(X)
    Y = np.cumsum(X)
    Y -= Ave
    L = np.floor(len(X) * 1 / (2 ** np.array(list(range(4, int(np.log2(len(X))) - 4)))))
    F = np.zeros(len(L))  # F(n) of different given box length n
    for i in range(0, len(L)):
        n = int(L[i])  # for each box length L[i]
        if n == 0:
            print("time series is too short while the box length is too big")
            print("abort")
            exit()
        for j in range(0, len(X), n):  # for each box
            if j + n < len(X):
                c = list(range(j, j + n))
                # coordinates of time in the box
                c = np.vstack([c, np.ones(n)]).T
                # the value of data in the box
                y = Y[j:j + n]
                # add residue in this box
                F[i] += np.linalg.lstsq(c, y)[1]
        F[i] /= ((len(X) / n) * n)
    F = np.sqrt(F)
    return np.linalg.lstsq(np.vstack([np.log(L), np.ones(len(L))]).T, np.log(F))[0][0]


def bin_power(X, Fs, Band=None):  # Fs = sampling freq of X
    if Band is None:
        Band = [0.5, 4, 7, 12, 30]
    C = np.fft.fft(X)
    C = abs(C)
    Power = np.zeros(len(Band) - 1)
    for Freq_Index in range(0, len(Band) - 1):
        Freq = float(Band[Freq_Index])
        Next_Freq = float(Band[Freq_Index + 1])
        Power[Freq_Index] = sum(C[int(np.floor(Freq / Fs * len(X))): int(np.floor(Next_Freq / Fs * len(X)))])
    Power_Ratio = Power / sum(Power)
    return Power_Ratio


def spectral_entropy(X, Power_Ratio):
    Spectral_Entropy = 0
    for i in range(0, len(Power_Ratio) - 1):
        Spectral_Entropy += Power_Ratio[i] * np.log(Power_Ratio[i])
    Spectral_Entropy /= np.log(len(Power_Ratio))  # to save time, minus one is omitted
    return -1 * Spectral_Entropy


def information_based_similarity(x, y, n=8):
    """Calculates the information based similarity of two time series x and y."""
    Wordlist = []
    Space = [[0, 0], [0, 1], [1, 0], [1, 1]]
    Sample = [0, 1]
    if n == 1:
        Wordlist = Sample
    if n == 2:
        Wordlist = Space
    elif n > 1:
        Wordlist = Space
        Buff = []
        for k in range(0, n - 2):
            Buff = [tuple(Wordlist[i]) for i in range(0, len(Wordlist))]
            Buff = tuple(Buff)
            Wordlist = []
            for i in range(0, len(Buff)):
                for j in range(0, len(Sample)):
                    Wordlist.append(list(Buff[i]))
                    Wordlist[len(Wordlist) - 1].append(Sample[j])
    Wordlist.sort()
    Input = [[], []]
    Input[0] = x
    Input[1] = y
    SymbolicSeq = [[], []]
    for i in range(0, 2):
        Encoder = np.diff(Input[i])
        for j in range(0, len(Input[i]) - 1):
            if Encoder[j] > 0:
                SymbolicSeq[i].append(1)
            else:
                SymbolicSeq[i].append(0)
    Wm = [embed_seq(SymbolicSeq[0], 1, n).tolist(), embed_seq(SymbolicSeq[1], 1, n).tolist()]
    Count = [[], []]
    for i in range(0, 2):
        for k in range(0, len(Wordlist)):
            Count[i].append(Wm[i].count(Wordlist[k]))
    Prob = [[], []]
    for i in range(0, 2):
        Sigma = 0
        for j in range(0, len(Wordlist)):
            Sigma += Count[i][j]
        for k in range(0, len(Wordlist)):
            Prob[i].append(np.true_divide(Count[i][k], Sigma))
    Entropy = [[], []]
    for i in range(0, 2):
        for k in range(0, len(Wordlist)):
            if Prob[i][k] == 0:
                Entropy[i].append(0)
            else:
                Entropy[i].append(Prob[i][k] * (np.log2(Prob[i][k])))
    Rank = [[], []]
    Buff = [[], []]
    Buff[0] = tuple(Count[0])
    Buff[1] = tuple(Count[1])
    for i in range(0, 2):
        Count[i].sort()
        Count[i].reverse()
        for k in range(0, len(Wordlist)):
            Rank[i].append(Count[i].index(Buff[i][k]))
            Count[i][Count[i].index(Buff[i][k])] = -1
    IBS = 0
    Z = 0
    n = 0
    for k in range(0, len(Wordlist)):
        if (Buff[0][k] != 0) & (Buff[1][k] != 0):
            F = -Entropy[0][k] - Entropy[1][k]
            IBS += np.multiply(np.absolute(Rank[0][k] - Rank[1][k]), F)
            Z += F
        else:
            n += 1
    IBS = np.true_divide(IBS, Z)
    IBS = np.true_divide(IBS, len(Wordlist) - n)
    return IBS


def hurst(X):
    N = X.size
    T = np.arange(1, N + 1)
    Y = np.cumsum(X)
    Ave_T = Y / T
    S_T = np.zeros(N)
    R_T = np.zeros(N)
    for i in range(N):
        S_T[i] = np.std(X[:i + 1])
        X_T = Y - T * Ave_T[i]
        R_T[i] = np.ptp(X_T[:i + 1])
    R_S = R_T / S_T
    R_S = np.log(R_S)[1:]
    n = np.log(T)[1:]
    A = np.column_stack((n, np.ones(n.size)))
    [m, c] = np.linalg.lstsq(A, R_S)[0]
    return m


def hjorth(X, D=None):
    if D is None:
        D = np.diff(X)
        D = D.tolist()
    D.insert(0, X[0])  # pad the first difference
    D = np.array(D)
    n = len(X)
    M2 = float(sum(D ** 2)) / n
    TP = sum(np.array(X) ** 2)
    M4 = 0
    for i in range(1, len(D)):
        M4 += (D[i] - D[i - 1]) ** 2
    M4 = M4 / n
    return np.sqrt(M2 / TP), np.sqrt(float(M4) * TP / M2 / M2)


def pfd(X, D=None):
    if D is None:
        D = np.diff(X)
        D = D.tolist()
    N_delta = 0  # number of sign changes in derivative of the signal
    for i in range(1, len(D)):
        if D[i] * D[i - 1] < 0:
            N_delta += 1
    n = len(X)
    return np.log10(n) / (np.log10(n) + np.log10(n / n + 0.4 * N_delta))


def Power(x):
    F = np.fft.fft(x, axis=-1)
    P = F * np.conjugate(F)
    return np.sum(P, axis=-1).real / x.shape[1]
