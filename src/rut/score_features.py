import numpy as np
import pandas as pd
from functools import partial, reduce
from contextlib import closing
from multiprocessing import Pool
from rut.misc import category_to_numeric
from rut.stats import confidence_interval
# from rut.sampling import draw_sample, find_sampling_value, normalize


def _score_features(samples, feature_sets):
    """score a downsampled group of observations against a set of features

    :param [np.ndarray(int)] samples: list of observations x genes samples
    :param [np.ndarray(int)] feature_sets: list of feature sets converted to numerical
      indices by score_features (main function)

    :return np.array: categories x feature_sets
    """

    # preallocate
    results = np.zeros((len(samples), len(feature_sets)), dtype=float)

    for i, arr in enumerate(samples):
        for j, fset in enumerate(feature_sets):
            results[i, j] = np.sum(arr[:, fset], axis=1).mean()

    return results


def score_features(
        feature_sets, data, labels, n_iter=50, sampling_percentile=10,
        upsample=False, max_obs_per_sample=500, processes=None):
    """
    score a group of observations for the contribution of a set of features, utilizing
    sampling to equalize the weight of each observation.

    :param dict feature_sets: a dictionary of one or more labeled feature sets to score
      against data
    :param pd.DataFrame data: observations x features matrix
    :param np.ndarray labels: data will be split into groups according to labels and
      each group will be scored against each feature_set
    :param int n_iter: number of times to sample observations
    :param int sampling_percentile: percentile in (1, 99) to downsample to. observations
      with row sums lower than this value will be excluded
    :param upsample: if False, observations with size lower than sampling_percentile are
      discarded. If True, those observations are upsampled (Not recommended).
    :param int max_obs_per_sample: Maximum number of observations to use in each sample
      useful to set ceiling on memory usage. Default=500
    :param int processes: Specify the number of processes to spawn. if not None, uses the
      maximum number of cores on your computer.

    :return np.array | pd.DataFrame: n observations x scores
    """

    # normalize the data and calculate split points
    features = data.columns
    data = data.values

    # if labels are not numeric, transform to numeric categories
    label_order = np.unique(labels)
    labels = category_to_numeric(labels)
    if not labels.shape[0] == data.shape[0]:
        raise ValueError('labels (shape=%s) must match dimension 0 of data (shape=%s)' %
                         (repr(labels.shape), repr(labels.data)))

    idx = np.argsort(labels)
    data = data[idx, :]  # will copy
    labels = labels[idx]

    splits = np.where(np.diff(labels))[0] + 1

    # calculate sampling values
    v = find_sampling_value(np.split(data, splits), sampling_percentile)
    norm_data, labels = normalize(data, v, upsample, labels)

    norm_splits = np.where(np.diff(labels))[0] + 1  # re-diff, norm_data causes loss

    n_observation = min(d.shape[0] for d in np.split(norm_data, norm_splits))
    n_observation = min(n_observation, max_obs_per_sample)  # check obs ceiling from param

    # map features to integer values
    merged_features = np.array(reduce(np.union1d, feature_sets.values()))

    # toss features not in data
    merged_features = merged_features[np.in1d(merged_features, features)]
    print(merged_features.shape)

    numerical_feature_sets = [
        np.where(np.in1d(merged_features, fset))[0] for fset in feature_sets.values()]

    retain_features = np.in1d(features, merged_features)

    print(retain_features.shape)
    print(retain_features.sum())

    split_data = np.split(norm_data, norm_splits)

    print(split_data[0].shape)
    sampling_iterator = (
        [draw_sample(d, n_observation)[:, retain_features] for d in split_data]
        for _ in np.arange(n_iter))

    process_function = partial(
        _score_features,
        feature_sets=numerical_feature_sets,
    )

    # consume iterator with imap_unordered
    with closing(Pool(processes=processes)) as pool:
        results = pool.imap_unordered(process_function, sampling_iterator)

    # reduce and return result
    results = np.stack(list(results), axis=2)
    ci = confidence_interval(results, axis=2)
    mu = np.mean(results, axis=2)
    df = pd.Panel(
        np.concatenate([mu[None, :], ci.T], axis=0),
        items=['mean', 'ci_low', 'ci_high'],
        major_axis=label_order,
        minor_axis = list(feature_sets.keys())
    )

    return df
