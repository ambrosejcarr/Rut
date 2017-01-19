from functools import partial
from multiprocessing import Pool
from contextlib import closing
from itertools import repeat
import numpy as np
import pandas as pd
from scipy.stats.mstats import count_tied_groups, rankdata
from scipy.stats.mstats import kruskalwallis as _kruskalwallis
from scipy.special import erfc
from statsmodels.sandbox.stats.multicomp import multipletests


def _mannwhitneyu(x, y, use_continuity=True):
    """
    Computes the Mann-Whitney statistic
    Missing values in `x` and/or `y` are discarded.

    :param np.ndarray x: Input, vector or observations x features matrix
    :param np.ndarray y: Input, vector or observations x features matrix. If matrix, 
        must have same number of features as x
    :param bool use_continuity: {True, False}, optional
        Whether a continuity correction (1/2.) should be taken into account.

    :return (float, float, float):
      statistic : float
        The Mann-Whitney statistic
      approx z : float
        The normal-approximated z-score for U.
      pvalue : float
        Approximate p-value assuming a normal distribution.
    """
    if x.ndim == 1 and y.ndim == 1:
        x, y = x[:, np.newaxis], y[:, np.newaxis]
    ranks = rankdata(np.concatenate([x, y]), axis=0)
    nx, ny = x.shape[0], y.shape[0]
    nt = nx + ny
    U = ranks[:nx].sum(0) - nx * (nx + 1) / 2.

    mu = (nx * ny) / 2.
    u = np.amin([U, nx*ny - U], axis=0)  # get smaller U by convention

    sigsq = np.ones(ranks.shape[1]) * (nt ** 3 - nt) / 12.

    for i in np.arange(len(sigsq)):
        ties = count_tied_groups(ranks[:, i])
        sigsq[i] -= np.sum(v * (k ** 3 - k) for (k, v) in ties.items()) / 12.
    sigsq *= nx * ny / float(nt * (nt - 1))

    if use_continuity:
        z = (U - 1 / 2. - mu) / np.sqrt(sigsq)
    else:
        z = (U - mu) / np.sqrt(sigsq)

    prob = erfc(abs(z) / np.sqrt(2))
    return np.vstack([u, z, prob]).T


def find_sampling_value(group_data, percentile):
    """identify the sampling value

    :param list group_data: among each group, the row sum at percentile is
      evaluated, and the minimum value across the groups is the sampling value 
    :param int percentile: the percentile to select for the sampling value

    :return int: the minimum value at percentile across the groups in group_data
    """
    return min(np.percentile(g.sum(axis=1), percentile) for g in group_data)


def normalize(data, downsample_value, upsample=False, labels=None):
    """Normalize data such that each observation sums to downsample_value

    This function has an important downstream interaction with _draw_sample. In draw
    sample, non-integer values are rounded probabilistically, such that integer outputs
    are produced this has the effect of producing observations with very similar
    distributions to their down-sampled observations, but with more uncertainty
    associated with observations that have sparser sampling

    :param np.ndarray data: array containing observations x features
    :param int downsample_value: value to normalize observation counts to. 
    :param bool upsample: if False, all observations with size < downsample_value are
      excluded. If True, those observations are upsampled to downsample_value.
    :return np.ndarray: array containing observations x samples where rows sum to
      downsample_value
    """
    obs_size = data.sum(axis=1)
    if not upsample:
        keep = obs_size >= downsample_value
        data = data[keep, :]
        if labels is not None:
            labels = labels[keep]
    norm = (data * downsample_value) / data.sum(axis=1)[:, np.newaxis]
    if labels is not None:
        return norm, labels
    else:
        return norm


def _draw_sample(normalized_data, n):
    """Randomly sample n normalized observations from normalized_data
    :param np.ndarray normalized_data: normalized observations x features matrix
    :param int n: number of observations to draw from normalized_data

    :return np.ndarray: n x features array
    """
    np.random.seed()
    idx = np.random.randint(0, normalized_data.shape[0], n)
    sample = normalized_data[idx, :]
    p = np.random.sample(sample.shape)  # round samples probabilistically

    return np.floor(sample) + (sample % 1 > p).astype(int)


def _mw_sampling_function(norm_data, n_observation):
    """Compute the Mann-Whitney U test on a single sample drawn from norm_data
    :param norm_data: normalized array containing observations from a column of x and 
       a column of y
    :param n_observation: the number of observations to draw from x and y

    :return (int, float, float): U, z, p-value, the result of _mannwhitneyu called on a 
       single sample (see _mannwhitneyu for more information on the structure of this 
       output
    """
    a, b = (_draw_sample(d, n_observation) for d in norm_data)
    return _mannwhitneyu(a, b)  # dim = (n_features, 3)


def confidence_interval(z):
    """Compute the 95% empirical confidence interval around z

    :param np.ndarray z: array containing the z-scores of each sampling run
    :return (int, int): 2.5th and 97.5th percentile z-scores
    """
    return np.percentile(z, [2.5, 97.5], axis=0).T


def mannwhitneyu(
        x, y, n_iter=50, sampling_percentile=10, alpha=0.05, verbose=False,
        upsample=False):
    """
    Compute a resampled Mann-Whitney U test between observations in each column (feature)
    of x and y. n_iter samples will be drawn from x and y, selecting a number of
    observations equal to the smaller of the two sets (array with smaller number of rows).
    all observations will be downsampled to the sampling_percentile before comparison to
    guarantee equivalent sampling.

    :param x: observations by features array or DataFrame (ndim must be 2, although there
      needn't be more than one feature)
    :param y: observations by features array or DataFrama. Features must be the same as x
    :param n_iter: number of times to sample x and y
    :param sampling_percentile: percentile to downsample to. observations with row sums
      lower than this value will be excluded
    :param alpha: significance threshold for FDR correction
    :param verbose: if True, report number of observations sampled in each iteration and
      the integer value to which observations are downsampled
    :param upsample: if False, observations with size lower than sampling_percentile are
      discarded. If True, those observations are upsampled.

    :return pd.DataFrame: DataFrame with columns corresponding to:
        U: median u-statistic over the n_iter iterations of the test
        z_approx: median approximate tie-corrected z-score for the mann-whitney U-test
        z_lo: lower bound, 95% confidence interval over z
        z_hi: upper bound, 95% confidence interval over z
        p: p-value for z_approx
        q: FDR-corrected q-value over all tests in output, using two-stage BH-FDR.
    """

    # do some sanity checks on input data
    if isinstance(x, pd.DataFrame) and isinstance(y, pd.DataFrame):
        assert np.array_equal(x.columns, y.columns)
        labels = x.columns
        x = x.values
        y = y.values
    elif x.ndim > 1:
        assert x.shape[1] == y.shape[1]
        labels = None
    else:
        labels = None

    # calculate sampling values
    v = find_sampling_value([x, y], sampling_percentile)
    norm_data = [normalize(d, v, upsample) for d in [x, y]]
    n_observation = min(d.shape[0] for d in norm_data)
    sampling_function = partial(_mw_sampling_function, n_observation=n_observation)

    if verbose:  # report sampling values
        print('sampling %d observations (with replacement) per iteration' % n_observation)
        print('sampling %d counts per observation' % v)

    with closing(Pool()) as pool:
        results = pool.map(sampling_function, repeat(norm_data, n_iter))

    results = np.stack(results)  # u, z, p

    ci = confidence_interval(results[:, :, 1])
    results = pd.DataFrame(
        data=np.concatenate([np.median(results, axis=0), ci], axis=1),
        index=labels,
        columns=['U', 'z_approx', 'p', 'z_lo', 'z_hi'])

    # add multiple-testing correction
    results['q'] = multipletests(results['p'], alpha=alpha, method='fdr_tsbh')[1]

    # remove low-value features whose median sampling value is -inf
    neginf = np.isneginf(results['z_approx'])
    results.ix[neginf, 'z_lo'] = np.nan
    results.ix[neginf, 'z_approx'] = 0
    results.ix[neginf, ['p', 'q']] = 1.

    results = results[['U', 'z_approx', 'z_lo', 'z_hi', 'p', 'q']].sort_values('q')
    results.iloc[:, 1:4] = np.round(results.iloc[:, 1:4], 2)

    return results


def _kw_sampling_function(data, splits, n_observation):
    """
    Draw subsamples of the observations from each partition of the normalized data

    :param np.ndarray data: normalized and sorted np.ndarray to draw samples from
    :param np.ndarray splits: indices that separate groups contained in the rows
      of data
    :param n_observation: number of observations to sample from each group

    :return list: result of the kruskal test on a single sampling draw (see _kruskal
      for details on the structure of the output) 
    """
    data = [_draw_sample(d, n_observation) for d in np.split(data, splits)]
    return _kruskal(data)


def _kruskal(data):
    """
    Compute the Kruskal-Wallis H-test for independent samples

    :param list data: list of lists of np.ndarrays to be submitted to kruskalwallis

    :return list: list of (H, p) the kruskalwallis statistic and resulting p-value
    """
    results = []
    for i in np.arange(data[0].shape[1]):
        args = [d[:, i] for d in data]
        try:
            results.append(_kruskalwallis(*args))
        except ValueError:
            results.append([0, 1.])
    return results


def category_to_numeric(labels):
    """transform categorical labels to a numeric array"""
    labels = np.array(labels)
    if np.issubdtype(labels.dtype, np.integer):
        return labels
    else:
        cats = np.unique(labels)
        map_ = dict(zip(cats, np.arange(cats.shape[0])))
        return np.array([map_[i] for i in labels])


def kruskalwallis(
        data, labels, n_iter=50, sampling_percentile=10, alpha=0.05, verbose=False,
        upsample=False):
    """Compute a resampled Kruskal-Wallis H-test for independent samples

    :param data: np.ndarray or pd.DataFrame of observations x features
    :param labels: observation labels for categories to be compared
    :param n_iter: number of times to sample x and y
    :param sampling_percentile: percentile to downsample to. observations with row sums
      lower than this value will be excluded
    :param alpha: significance threshold for FDR correction
    :param verbose: if True, report number of observations sampled in each iteration and
      the integer value to which observations are downsampled
    :param upsample: if False, observations with size lower than sampling_percentile are
      discarded. If True, those observations are upsampled.
    :return pd.DataFrame: DataFrame with columns:
      H: median u-statistic over the n_iter iterations of the test
      z_approx: median approximate tie-corrected z-score for the mann-whitney U-test
      z_lo: lower bound, 95% confidence interval over z
      z_hi: upper bound, 95% confidence interval over z
      p: p-value for z_approx
      q: FDR-corrected q-value over all tests in output, using two-stage BH-FDR.
    """

    if isinstance(data, pd.DataFrame):
        features = data.columns
        data = data.values
    elif isinstance(data, np.ndarray):
        features = None
    else:
        raise ValueError('data must be a np.ndarray or pd.DataFrame, not %s' %
                         repr(type(data)))

    # if labels are not numeric, transform to numeric categories
    labels = category_to_numeric(labels)
    if not labels.shape[0] == data.shape[0]:
        raise ValueError('labels (shape=%s) must match dimension 0 of data (shape=%s)' %
                         (repr(labels.shape), repr(labels.data)))

    idx = np.argsort(labels)
    data = data[idx, :]  # will copy
    labels = labels[idx]

    splits = np.where(np.diff(labels))[0] + 1

    # calculate sampling values and downsample data
    v = find_sampling_value(np.split(data, splits), sampling_percentile)
    norm_data, labels = normalize(data, v, upsample, labels)

    splits = np.where(np.diff(labels))[0] + 1  # rediff, norm_data causes loss

    n_observation = min(d.shape[0] for d in np.split(norm_data, splits))
    sampling_function = partial(_kw_sampling_function, n_observation=n_observation,
                                splits=splits)

    if verbose:  # report sampling values
        print('sampling %d observations (with replacement) per iteration' % n_observation)
        print('sampling %d counts per observation' % v)

    with closing(Pool()) as pool:
        results = pool.map(sampling_function, repeat(norm_data, n_iter))

    results = np.stack(results)  # H, p

    ci = confidence_interval(results[:, :, 0])  # around H
    results = pd.DataFrame(
        data=np.concatenate([np.median(results, axis=0), ci], axis=1),
        index=features,
        columns=['H', 'p', 'H_lo', 'H_hi'])

    results['q'] = multipletests(results['p'], alpha=alpha, method='fdr_tsbh')[1]
    results = results[['H', 'H_lo', 'H_hi', 'p', 'q']]
    return results
