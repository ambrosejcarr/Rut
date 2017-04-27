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
