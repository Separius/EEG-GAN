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


def pfd(X, D=None):
    """Compute Petrosian Fractal Dimension of a time series from either two
    cases below:
        1. X, the time series of type list (default)
        2. D, the first order differential sequence of X (if D is provided,
           recommended to speed up)
    In case 1, D is computed using Numpy's difference function.
    To speed up, it is recommended to compute D before calling this function
    because D may also be used by other functions whereas computing it here
    again will slow down.
    """
    if D is None:
        D = np.diff(X)
        D = D.tolist()
    N_delta = 0  # number of sign changes in derivative of the signal
    for i in range(1, len(D)):
        if D[i] * D[i - 1] < 0:
            N_delta += 1
    n = len(X)
    return np.log10(n) / (
            np.log10(n) + np.log10(n / n + 0.4 * N_delta)
    )


def hfd(X, Kmax):
    """ Compute Hjorth Fractal Dimension of a time series X, kmax
     is an HFD parameter
    """
    L = []
    x = []
    N = len(X)
    for k in range(1, Kmax):
        Lk = []
        for m in range(0, k):
            Lmk = 0
            for i in range(1, int(np.floor((N - m) / k))):
                Lmk += abs(X[m + i * k] - X[m + i * k - k])
            Lmk = Lmk * (N - 1) / np.floor((N - m) / float(k)) / k
            Lk.append(Lmk)
        L.append(np.log(np.mean(Lk)))
        x.append([np.log(float(1) / k), 1])

    (p, r1, r2, s) = np.linalg.lstsq(x, L)
    return p[0]


def information_based_similarity(x, y, n):
    """Calculates the information based similarity of two time series x
    and y.
    Parameters
    ----------
    x
        list
        a time series
    y
        list
        a time series
    n
        integer
        word order
    Returns
    ----------
    IBS
        float
        Information based similarity
    Notes
    ----------
    Information based similarity is a measure of dissimilarity between
    two time series. Let the sequences be x and y. Each sequence is first
    replaced by its first ordered difference(Encoder). Calculating the
    Heaviside of the resulting sequences, we get two binary sequences,
    SymbolicSeq. Using PyEEG function, embed_seq, with lag of 1 and dimension
    of n, we build an embedding matrix from the latter sequence.
    Each row of this embedding matrix is called a word. Information based
    similarity measures the distance between two sequence by comparing the
    rank of words in the sequences; more explicitly, the distance, D, is
    calculated using the formula:
    "1/2^(n-1) * sum( abs(Rank(0)(k)-R(1)(k)) * F(k) )" where Rank(0)(k)
    and Rank(1)(k) are the rank of the k-th word in each of the input
    sequences. F(k) is a modified "shannon" weighing function that increases
    the weight of each word in the calculations when they are more frequent in
    the sequences.
    It is advisable to calculate IBS for numerical sequences using 8-tupple
    words.
    References
    ----------
    Yang AC, Hseu SS, Yien HW, Goldberger AL, Peng CK: Linguistic analysis of
    the human heartbeat using frequency and rank order statistics. Phys Rev
    Lett 2003, 90: 108103
    Examples
    ----------
    >>> import pyeeg
    >>> from np.random import randn
    >>> x = randn(100)
    >>> y = randn(100)
    >>> pyeeg.information_based_similarity(x,y,8)
    0.64512947848249214
    """

    Wordlist = []
    Space = [[0, 0], [0, 1], [1, 0], [1, 1]]
    Sample = [0, 1]

    if (n == 1):
        Wordlist = Sample

    if (n == 2):
        Wordlist = Space

    elif (n > 1):
        Wordlist = Space
        Buff = []
        for k in range(0, n - 2):
            Buff = []

            for i in range(0, len(Wordlist)):
                Buff.append(tuple(Wordlist[i]))
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
            if (Encoder[j] > 0):
                SymbolicSeq[i].append(1)
            else:
                SymbolicSeq[i].append(0)

    Wm = []
    Wm.append(embed_seq(SymbolicSeq[0], 1, n).tolist())
    Wm.append(embed_seq(SymbolicSeq[1], 1, n).tolist())

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
            if (Prob[i][k] == 0):
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
        if ((Buff[0][k] != 0) & (Buff[1][k] != 0)):
            F = -Entropy[0][k] - Entropy[1][k]
            IBS += np.multiply(np.absolute(Rank[0][k] - Rank[1][k]), F)
            Z += F
        else:
            n += 1

    IBS = np.true_divide(IBS, Z)
    IBS = np.true_divide(IBS, len(Wordlist) - n)

    return IBS
