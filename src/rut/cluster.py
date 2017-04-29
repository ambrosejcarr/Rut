import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import phenograph
# from rut.sampling import normalize, round_random

# todo | parallelize sampling if memory usage allows.


def cluster(data, n_iter=10, sampling_percentile=10):
    """

    :param np.array data:
    :param int n_iter:
    :param int sampling_percentile: percentile to down-sample to
    :return np.array: cluster labels
    """

    sampling_value = np.percentile(data.sum(axis=1), sampling_percentile)

    # normalize to sampling value
    normalized, labels = normalize(
        data, sampling_value, labels=np.arange(data.shape[0]))

    # cluster n_iter different samples from data
    clustering_results = []
    for i in np.arange(n_iter):

        # sample
        normalized_rounded = round_random(normalized)

        # pca and project
        pca = PCA(n_components=50, svd_solver='randomized')
        reduced = pca.fit_transform(normalized_rounded)

        # cluster
        clusters, *_ = phenograph.cluster(reduced)
        clustering_results.append(clusters)

    # integrate results into a similarity matrix
    integrated = np.zeros((normalized.shape[0], normalized.shape[0]), dtype=np.int)
    for i in np.arange(n_iter):
        labs = clustering_results[i]
        for j in np.arange(integrated.shape[0]):
            increment = labs == labs[j]
            integrated[j, :] += increment.astype(int)

    # get mean cluster size
    k = int(np.round(np.mean([np.unique(r).shape[0] for r in clustering_results])))

    # cluster the integrated similarity matrix
    km = KMeans(n_clusters=15)
    metacluster_labels = km.fit_predict(integrated)

    # propagate ids back to original coordinates and return
    cluster_ids = np.ones(data.shape[0], dtype=np.int) * -1
    cluster_ids[labels] = metacluster_labels
    return cluster_ids
