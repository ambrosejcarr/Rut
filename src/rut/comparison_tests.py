import numpy as np
import pandas as pd
from scipy.special import comb
import multiprocessing
from contextlib import closing
from functools import partial
from statsmodels.sandbox.stats.multicomp import multipletests


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

