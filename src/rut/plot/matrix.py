import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import leaves_list
from fastcluster import linkage


def preordered_matrix(data, trim=2, ax=None, **kwargs):
    vmin = np.percentile(data, trim)
    vmax = np.percentile(data, 100 - trim)

    if ax is None:
        _, ax = plt.subplots(figsize=(3.5, 3))
    mesh = ax.pcolorfast(data, vmin=vmin, vmax=vmax, **kwargs)
    plt.colorbar(mappable=mesh, ax=ax)
    ax.set_xlabel('cells')
    ax.set_ylabel('cells')
    ax.set_title('co-clustering frequency')
    plt.tight_layout()
    return ax


def clustered_symmetric_matrix(data, trim=2, ax=None, **kwargs):
    """

    :param pd.DataFrame data:
    :param trim:
    :param ax:
    :param kwargs:
    :return:
    """
    z, order = _cluster_dimension(data)
    ordered = data.iloc[:, order].iloc[order, :]
    preordered_matrix(data=ordered, trim=trim, ax=ax, **kwargs)
    vmin = np.percentile(ordered, trim)
    vmax = np.percentile(ordered, 100 - trim)

    if ax is None:
        _, ax = plt.subplots(figsize=(3.5, 3))
    mesh = ax.pcolorfast(ordered, vmin=vmin, vmax=vmax, **kwargs)
    cb = plt.colorbar(mappable=mesh, ax=ax)
    ax.set_xlabel('cells')
    ax.set_ylabel('cells')
    ax.set_title('co-clustering frequency')
    plt.tight_layout()
    return ax


def _cluster_dimension(data, axis=1, method='single'):
    if axis == 0:
        data = data.T  # cluster the observations
    z = linkage(data, method=method)
    order = leaves_list(z)
    return z, order
