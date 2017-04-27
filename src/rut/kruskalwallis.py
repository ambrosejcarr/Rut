from contextlib import closing
from multiprocessing import Pool
import numpy as np
import pandas as pd
from scipy.stats.mstats import kruskalwallis as _kruskalwallis
from statsmodels.sandbox.stats.multicomp import multipletests
from rut.sampling import draw_sample, find_sampling_value, normalize
from rut.stats import confidence_interval
from rut.misc import category_to_numeric


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
    data = [draw_sample(d, n_observation) for d in np.split(data, splits)]
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


def kruskalwallis(
        data, labels, n_iter=50, sampling_percentile=10, alpha=0.05, verbose=False,
        upsample=False, max_obs_per_sample=500, processes=None):
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
      discarded. If True, those observations are upsampled (Not recommended).
    :param int max_obs_per_sample: Maximum number of observations to use in each sample
      useful to set ceiling on memory usage. Default=500
    :param int processes: Specify the number of processes to spawn. if not None, uses the
      maximum number of cores on your computer.


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

    # calculate sampling values and down-sample data
    v = find_sampling_value(np.split(data, splits), sampling_percentile)
    norm_data, labels = normalize(data, v, upsample, labels)

    norm_splits = np.where(np.diff(labels))[0] + 1  # re-diff, norm_data causes loss

    n_observation = min(d.shape[0] for d in np.split(norm_data, norm_splits))
    n_observation = min(n_observation, max_obs_per_sample)  # check obs ceiling from param

    if verbose:  # report sampling values
        print('sampling %d observations (with replacement) per iteration' % n_observation)
        print('sampling %d counts per observation' % v)

    # draw samples and shunt them to the kruskal test to save memory overhead
    # (3s sampling vs 40s kruskal)
    # create an iterator to draw samples
    split_data = np.split(norm_data, norm_splits)
    sampling_iterator = (
        [draw_sample(d, n_observation) for d in split_data] for _ in np.arange(n_iter))

    # consume iterator with imap_unordered
    with closing(Pool(processes=processes)) as pool:
        results = pool.imap_unordered(_kruskal, sampling_iterator)

    results = np.stack(results)  # H, p

    ci = confidence_interval(results[:, :, 0])  # around H
    results = pd.DataFrame(
        data=np.concatenate([np.median(results, axis=0), ci], axis=1),
        index=features,
        columns=['H', 'p', 'H_lo', 'H_hi'])

    results['q'] = multipletests(results['p'], alpha=alpha, method='fdr_tsbh')[1]
    results = results[['H', 'H_lo', 'H_hi', 'p', 'q']]
    return results
