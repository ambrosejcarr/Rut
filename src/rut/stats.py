from scipy.special import erfc
import numpy as np


def z_to_p(z):
    """convert a z-score to a p-value

    :param np.ndarray | pd.DataFrame z: array of z-scores
    :return np.ndarray | pd.DataFrame: array of p-values in same shape as input data
    """
    return erfc(abs(z) / np.sqrt(2))


def confidence_interval(v, axis=0):
    """Compute the 95% empirical confidence interval around vector v

    :param np.ndarray v: array containing the values across sampling runs
    :return (int, int): 2.5th and 97.5th percentile of the passed values
    """
    return np.percentile(v, [2.5, 97.5], axis=axis).T


