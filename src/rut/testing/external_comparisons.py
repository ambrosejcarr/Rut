import numpy as np
import pandas as pd
from scipy.special import comb
import multiprocessing
from contextlib import closing
from functools import partial
from statsmodels.sandbox.stats.multicomp import multipletests
from scipy.stats.mstats import mannwhitneyu


def _binomial_test(gamma, Ng, N, n_choose_k):
    """

    :param int gamma:
    :param int Ng:
    :param int N:
    :param np.ndarray n_choose_k:
    :return:
    """
    nck = n_choose_k[Ng:N]
    k = np.arange(Ng, N)
    return np.sum(nck * (gamma ** k) * (1 - gamma) ** (N - k))


def binomial_test(a, b):
    """compute the 1-tailed p-value that a is greater than b

    :param np.ndarray a: m cells x n genes input data
    :param np.ndarray b: p cells x n genes input data
    :return np.ndarray p: vector of p-values
    :return np.ndarray q: vector of fdr-corrected q-values
    """

    # sum over k = Ng (slow?)

    M_g = np.maximum(np.sum(b > 0, axis=0), 1)  # pseudocount for gamma=0
    N_g = np.sum(a > 0, axis=0)
    M = b.shape[0]
    N = a.shape[0]
    gamma = M_g / M

    # compute this vector just once?
    n_choose_k = np.array([comb(N, k) for k in np.arange(N)])

    iterable = zip(gamma, N_g)

    test_function = partial(_binomial_test, N=N, n_choose_k=n_choose_k)
    with closing(multiprocessing.Pool()) as pool:
        p = pool.starmap(test_function, iterable, chunksize=1)

    return np.array(p)


def kartik_test(a, b):
    """carry out Kartik's binomial test, first testing a > b, then b > a

    :param pd.DataFrame a:
    :param pd.DataFrame b:
    :return pd.DataFrame:
    """

    p_a = binomial_test(a.values, b.values)
    p_b = binomial_test(b.values, a.values)
    direction = np.zeros_like(p_a)
    direction[p_a < p_b] = 1
    direction[p_a > p_b] = -1
    p = np.minimum(p_a, p_b)
    q = multipletests(p, method='fdr_bh')[1]
    return pd.DataFrame(
        {'direction': direction,
         'p': p,
         'q': q}, index=a.columns)


class BinomialTest:

    def __init__(self, data, labels):
        """
        Carry out a binomial test on data between groups given by labels

        :param pd.DataFrame data:
        :param pd.Series | np.ndarray labels:
        """

        self.data = data
        self.labels = labels
        if np.unique(labels).shape[0] != 2:
            raise ValueError('Binomial Test can only be applied to two groups, please use kruskalwallis for multi-'
                             'group comparisons')

    def fit(self):
        a, b = np.unique(self.labels)
        is_a = self.labels == a
        is_b = self.labels == b
        return kartik_test(self.data.loc[is_a, :], self.data.loc[is_b, :])


class MannWhitneyU:

    def __init__(self, data, labels):
        # library size normalize data
        self.data = data.div(data.sum(axis=1), axis=0)
        self.labels = labels
        if np.unique(labels).shape[0] != 2:
            raise ValueError('Mann Whitney U can only be applied to two groups')

    def fit(self):
        a, b = np.unique(self.labels)
        is_a = self.labels == a
        is_b = self.labels == b
        results = []
        for c in self.data.columns:
            results.append(mannwhitneyu(self.data.loc[is_a, c], self.data.loc[is_b, c]))

        return pd.DataFrame(np.vstack(results), index=self.data.columns, columns=['U', 'p'])
