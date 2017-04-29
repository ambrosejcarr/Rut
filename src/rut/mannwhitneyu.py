import warnings
from functools import partial
from multiprocessing import Pool
from contextlib import closing
from itertools import repeat
import numpy as np
import pandas as pd
from scipy.stats.mstats import count_tied_groups, rankdata
from statsmodels.sandbox.stats.multicomp import multipletests
# from rut.sampling import draw_sample, normalize, find_sampling_value
from rut.stats import z_to_p, confidence_interval


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

    # calculate U for n1
    if x.ndim == 1 and y.ndim == 1:
        x, y = x[:, np.newaxis], y[:, np.newaxis]
    ranks = rankdata(np.concatenate([x, y]), axis=0)
    nx, ny = x.shape[0], y.shape[0]
    nt = nx + ny
    U = ranks[:nx].sum(0) - nx * (nx + 1) / 2.

    # get mean value
    mu = (nx * ny) / 2.

    # get smaller u (convention) for reporting only
    u = np.amin([U, nx*ny - U], axis=0)

    sigsq = np.ones(ranks.shape[1]) * (nt ** 3 - nt) / 12.

    for i in np.arange(len(sigsq)):
        ties = count_tied_groups(ranks[:, i])
        sigsq[i] -= np.sum(v * (k ** 3 - k) for (k, v) in ties.items()) / 12.
    sigsq *= nx * ny / float(nt * (nt - 1))

    # ignore division by zero warnings; they are properly dealt with by this test.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if use_continuity:
            # note that in the case that sigsq is zero, and mu=U, this line will produce
            # -inf. However, this z-score should be zero, as this is an artifact of
            # continuity correction
            z = (U - 1 / 2. - mu) / np.sqrt(sigsq)
        else:
            z = (U - mu) / np.sqrt(sigsq)

    # correct infs
    z[sigsq == 0] = 0

    prob = z_to_p(z)
    return np.vstack([u, z, prob]).T


def _mw_sampling_function(norm_data, n_observation):
    """Compute the Mann-Whitney U test on a single sample drawn from norm_data
    :param norm_data: normalized array containing observations from a column of x and
       a column of y
    :param n_observation: the number of observations to draw from x and y

    :return (int, float, float): U, z, p-value, the result of _mannwhitneyu called on a
       single sample (see _mannwhitneyu for more information on the structure of this
       output
    """
    a, b = (draw_sample(d, n_observation) for d in norm_data)
    return _mannwhitneyu(a, b)  # dim = (n_features, 3)


def mannwhitneyu(
        x, y, n_iter=50, sampling_percentile=10, alpha=0.05, verbose=False,
        upsample=False, max_obs_per_sample=500, report_z_distributions=False,
        expected_value_function=np.median, processes=None):
    """
    Compute a resampled Mann-Whitney U test between observations in each column (feature)
    of x and y. n_iter samples will be drawn from x and y, selecting a number of
    observations equal to the smaller of the two sets (array with smaller number of rows).
    all observations will be downsampled to the sampling_percentile before comparison to
    guarantee equivalent sampling.

    :param pd.DataFrame | np.ndarray x: observations by features array or DataFrame
      (n-dim must be 2, although there needn't be more than one feature)
    :param pd.DataFrame | np.ndarray y: observations by features array or DataFrame.
      Features must be the same as x
    :param int n_iter: number of times to sample x and y
    :param int sampling_percentile: percentile to down-sample to. observations with row
      sums lower than this value will be excluded
    :param float alpha: significance threshold for FDR correction
    :param bool verbose: if True, report number of observations sampled in each iteration
      and the integer value to which observations are down-sampled
    :param bool upsample: if False, observations with size lower than sampling_percentile
      are discarded. If True, those observations are up-sampled.
    :param int max_obs_per_sample: Maximum number of observations to use in each sample
      useful to set ceiling on memory usage. Default=500
    :param bool report_z_distributions: if True, a second DataFrame of shape n genes x p
      iterations with entries equal to the number of iterations will be reported
    :param function expected_value_function: function to extract the expected value from a
      list of values. Default = np.median, but np.mean may also be used.
    :param int processes: Specify the number of processes to spawn. if not None, uses the
      maximum number of cores on your computer.

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
    n_observation = min(n_observation, max_obs_per_sample)  # check obs ceiling from param
    sampling_function = partial(_mw_sampling_function, n_observation=n_observation)

    if verbose:  # report sampling values
        print('sampling %d observations (with replacement) per iteration' % n_observation)
        print('sampling %d counts per observation' % v)

    with closing(Pool(processes=processes)) as pool:
        results = pool.map(sampling_function, repeat(norm_data, n_iter))

    # todo examine memory impact of removing overwriting of results (reason: clarity)
    results = np.stack(results)  # (u, z, p) x n_iter

    ci = confidence_interval(results[:, :, 1])

    if report_z_distributions:
        zdist = pd.DataFrame(
            data=results[:, :, 1].T,
            index=labels,
        )
        zdist.columns.name = 'iteration number'

    results = pd.DataFrame(
        data=np.concatenate([expected_value_function(results, axis=0), ci], axis=1),
        index=labels,
        columns=['U', 'z_approx', 'p', 'z_lo', 'z_hi'])

    # add multiple-testing correction
    results['q'] = multipletests(results['p'], alpha=alpha, method='fdr_tsbh')[1]

    results = results[['U', 'z_approx', 'z_lo', 'z_hi', 'p', 'q']].sort_values('q')
    results.iloc[:, 1:4] = np.round(results.iloc[:, 1:4], 2)

    if report_z_distributions:
        return results, zdist
    else:
        return results
