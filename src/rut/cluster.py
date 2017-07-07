import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import phenograph
from rut import sample, misc


class Cluster(sample.Sampled):

    def __init__(self, *args, **kwargs):
        self.cluster_labels_ = None
        self.confusion_matrix_ = None
        self.association_strength_ = None
        super().__init__(*args, **kwargs)

    @classmethod
    def map(cls, n_jobs=1, *args):
        """
        calculate phenograph community assignments across resampled data for each
        of the n observations in global variable data

        :param args: allows passing of arbitrary arguments necessary to evoke
          multiprocessing.Pool.map()
        :param n_jobs: Default=1, construct kernel in parallel
        :return np.ndarray: (n,) cluster assignments
        """
        complete_data = cls.get_shared_data()
        sample = cls._draw_sample(complete_data)
        pca = PCA(n_components=50, svd_solver='randomized')
        reduced = pca.fit_transform(sample)

        # cluster, omitting phenograph outputs
        with misc.suppress_stdout_stderr():
            clusters, *_ = phenograph.cluster(reduced, n_jobs=n_jobs)
        return clusters

    # todo consider building this as a sparse matrix (it probably is!)
    def reduce(self, results, save_confusion_matrix):
        """
        carry out consensus clustering across iterations of phenograph run on data,
        running k-means on the confusion matrix generated by these iterations with k fixed
        as the median number of communities selected by phenograph.

        :param list results: list of vectors of cluster labels
        :param bool save_confusion_matrix: if True, the confusion matrix generated across
          in-drop runs will be saved.
        :return np.ndarray: (n,) consensus cluster labels
        """
        # create confusion matrix
        n = self._index.shape[0]
        integrated = np.zeros((n, n), dtype=np.uint8)
        for i in np.arange(len(results)):
            labs = results[i]
            for j in np.arange(integrated.shape[0]):
                increment = np.equal(labs, labs[j]).astype(np.uint8)
                integrated[j, :] += increment

        # get mean cluster size
        k = int(np.round(np.mean([np.unique(r).shape[0] for r in results])))

        # cluster the integrated similarity matrix
        km = KMeans(n_clusters=15)
        metacluster_labels = km.fit_predict(integrated)

        # propagate ids back to original coordinates and return
        cluster_ids = np.ones(n, dtype=np.int) * -1
        cluster_ids[self._labels] = metacluster_labels

        if save_confusion_matrix:
            self.confusion_matrix_ = pd.DataFrame(
                integrated, index=cluster_ids, columns=cluster_ids)

        return cluster_ids

    # todo note that this crashes some threads on python 3.5; upgrade to python 3.6!?
    def fit(self, n_iter=10, n_processes=None, save_confusion_matrix=True):
        """fit consensus clustering across n_iter phenograph runs

        :param int n_iter: number of iterations of clustering to run
        :param int n_processes: number of processes to assign to clustering
        :param bool save_confusion_matrix: if True, save the confusion matrix generated
          by repeated phenograph runs
        :return np.array: self.cluster_labels_, consensus assignments across phenograph
          runs.
        """

        cluster_labels_ = self.run(
            n_iter=n_iter,
            fmap=self.map,
            freduce=self.reduce,
            n_processes=n_processes,
            freduce_kwargs=dict(save_confusion_matrix=save_confusion_matrix))

        self.cluster_labels_ = pd.Series(cluster_labels_, index=self._index)

        return self.cluster_labels_

    def fit_no_parallelism(self, n_iter, n_processes=1, save_confusion_matrix=True):
        """
        Fit consensus clustering across n_iter phenograph runs. Note that unless
        n_processes is 1, phenograph is still parallelized.

        :param int n_iter: number of iterations of clustering to run
        :param int n_processes: number of processes to assign to phenograph for kernel
          calculation
        :param bool save_confusion_matrix: if True, save the confusion matrix generated
          by repeated phenograph runs
        :return np.array: self.cluster_labels_, consensus assignments across phenograph
          runs.
        """

        results = []
        self._proc_init()
        for _ in np.arange(n_iter):
            results.append(self.map(n_jobs=n_processes))
        self.cluster_labels_ = self.reduce(
            results, save_confusion_matrix=save_confusion_matrix)

        return self.cluster_labels_

    # todo document me
    def plot_confusion_matrix(self, trim=2, ax=None, **kwargs):
        """

        :param trim:
        :param ax:
        :param kwargs:
        :return:
        """
        import rut.plot.matrix
        return rut.plot.matrix.clustered_symmetric_matrix(
            self.confusion_matrix_, trim=trim, ax=ax, **kwargs)

    # todo transform into property (?)
    # todo turn this into a confidence instead of just a co-occurrence frequency
    def score_association_strength(self, n_iter, plot=True):
        """
        in the future will return a confidence, currently returns frequency of
        association with own cluster

        :return np.ndarray: self.association_strength_
        """
        if self.confusion_matrix_ is None:
            raise ValueError('Fit the clustering with save_confusion_matrix=True before '
                             'estimating association strength')

        # for each cluster of size n, the maximum self-association is n**2. Can also
        # model cross-association, using groupby operations and sums, followed by sqrt

        group_sums = self.confusion_matrix_.groupby(level=0, axis=0).sum()
        group_sums = group_sums.groupby(level=0, axis=1).sum()
        all_labels, counts = np.unique(self.cluster_labels_, return_counts=True)
        association_strength = (
            group_sums.loc[all_labels, all_labels] / (counts[:, None] * counts[None, :]))
        association_strength /= n_iter
        association_strength.index.name = 'clusters'
        association_strength.columns.name = 'clusters'
        self.association_strength_ = association_strength
        if plot:
            import rut.plot.matrix
            ax = rut.plot.matrix.clustered_symmetric_matrix(
                self.association_strength_, trim=0)
        return self.association_strength_
