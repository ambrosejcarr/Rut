import os
from scipy.special import erfc
import numpy as np


# todo think about adding a return_order parameter to this function (returns cats)
def category_to_numeric(labels):
    """transform categorical labels to a numeric array

    :param labels: ordered iterable of labels
    :return np.ndarray: labels, transformed into numerical values
    """
    labels = np.array(labels)
    if np.issubdtype(labels.dtype, np.integer):
        return labels
    else:
        cats = np.unique(labels)
        map_ = dict(zip(cats, np.arange(cats.shape[0])))
        return np.array([map_[i] for i in labels])


class suppress_stdout_stderr(object):
    """
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    """

    def __init__(self):
        # Open a pair of null files
        self.null_fds =  [os.open(os.devnull,os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0],1)
        os.dup2(self.null_fds[1],2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0],1)
        os.dup2(self.save_fds[1],2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])


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
