import numpy as np

# todo | this module should define a class that acts as a generator to
# todo | draw infinite samples from a normalized data matrix


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
    :param np.ndarray labels: array or series of labels, useful to track which cells
      are being discarded by exclusion of observations with sums below the downsampling
      value
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


def round_random(sample):
    """round a sample probabilistically"""
    p = np.random.sample(sample.shape)
    return np.floor(sample) + (sample % 1 > p).astype(int)


def draw_sample(normalized_data, n, return_indices=False):
    """Randomly sample n normalized observations from normalized_data
    :param np.ndarray normalized_data: normalized observations x features matrix. Note
      that the sum of the features for each observation in this data determines the
      total number of times the features are observed. Hence, this parameter cannot
      be set in this function, but must be set by normalize()
    :param int n: number of observations to draw from normalized_data
    :param bool return_indices: if True, indices of sampled cells will also be returned

    :return np.ndarray: n x features array
    """
    np.random.seed()
    # select cells
    idx = np.random.randint(0, normalized_data.shape[0], n)
    sample = normalized_data[idx, :]

    sample = round_random(sample)
    if return_indices:
        return sample, idx
    else:
        return sample
