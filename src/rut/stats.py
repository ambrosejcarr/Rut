from scipy.special import erfc
import numpy as np


def z_to_p(z):
    """convert a z-score to a p-value

    :param np.ndarray | pd.DataFrame z: array of z-scores
    :return np.ndarray | pd.DataFrame: array of p-values in same shape as input data
    """
    return erfc(abs(z) / np.sqrt(2))


def confidence_interval(z):
    """Compute the 95% empirical confidence interval around z

    :param np.ndarray z: array containing the z-scores of each sampling run
    :return (int, int): 2.5th and 97.5th percentile z-scores
    """
    return np.percentile(z, [2.5, 97.5], axis=0).T


