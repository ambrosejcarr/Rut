from contextlib import closing
from multiprocessing import Pool, sharedctypes
import numpy as np
import pandas as pd
from scipy.stats.mstats import count_tied_groups, rankdata
import warnings
from rut.stats import z_to_p, confidence_interval
from rut.misc import suppress_stdout_stderr
from statsmodels.sandbox.stats.multicomp import multipletests
from scipy.stats.mstats import kruskalwallis as _kruskalwallis
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import phenograph
from functools import reduce, partial

# must test with openblas enabled and disabled
# see comment #3 on answer #3:
# http://stackoverflow.com/questions/17785275/share-large-read-only-numpy-array-between-multiprocessing-processes
# export OPENBLAS_NUM_THREADS=1


class Sampled:

    def __init__(
            self, data, labels, sampling_percentile=10, feature_sets=None, upsample=False,
            is_sorted=False):
        """

        :param pd.DataFrame | np.array data: m observations x p features array or
          dataframe
        :param np.ndarray labels:  condition labels that separate cells into units of
          comparison
        :param sampling_percentile:  the percentile of observation size to use as the
          sampling size of drawn samples
        :param dict feature_sets: (Optional) dictionary of iterables containing features.
          If provided, at least one feature per set must overlap with features in data for
          the set to be tested against the data.
        :param bool upsample: if True, observation sizes are equalized at the sampling
          value identified by sampling percentile, with smaller observations being
          upsampled to this level and no observations being discarded.
        :param bool is_sorted: if True, no sorting is done of data or labels
        """

        # check input types
        if len(labels) != data.shape[0]:
            raise ValueError('number of labels %d != number of observations %d'
                             % (len(labels), data.shape[0]))

        if isinstance(data, pd.DataFrame):
            index = data.index
            self._features = data.columns
            data = data.values
        elif isinstance(data, np.ndarray):
            index = np.arange(data.shape[0])
            self._features = np.arange(data.shape[1])
        else:
            raise TypeError('Data must be provided as a numpy array or pandas DataFrame, '
                            'not %s.' % repr(type(data)))

        # construct numeric labels
        numerical_label_order = np.unique(labels)
        map_ = dict(zip(numerical_label_order, np.arange(numerical_label_order.shape[0])))
        numeric_labels = np.array([map_[i] for i in labels])

        # sort data unless it was passed sorted
        if not is_sorted:
            idx = np.argsort(numeric_labels)
            data = data[idx, :]
            numeric_labels = numeric_labels[idx]
            labels = labels[idx]
            index = index[idx]

        # get first set of splits
        splits = np.where(np.diff(numeric_labels))[0] + 1

        # get observation sizes
        observation_sizes = data.sum(axis=1)

        # get sample size
        n_observations_per_sample_draw = min(
            np.percentile(g, sampling_percentile)
            for g in np.split(observation_sizes, splits))

        # throw out cells below sampling threshold unless upsampling is requested
        if not upsample:
            keep = observation_sizes >= n_observations_per_sample_draw
            data = data[keep, :]
            numeric_labels = numeric_labels[keep]
            labels = labels[keep]
            index = index[keep]
            observation_sizes = observation_sizes[keep]
            splits = np.where(np.diff(numeric_labels))[0] + 1
        self.n_observations_per_sample = min(
            g.shape[0] for g in np.split(observation_sizes, splits))

        # normalize data
        data = (data * n_observations_per_sample_draw) / observation_sizes[:, np.newaxis]

        # convert to shared array & expose data
        self.data = np.ctypeslib.as_ctypes(data)
        self.data = sharedctypes.Array(self.data._type_, self.data, lock=False)

        # expose splits
        self.splits = np.ctypeslib.as_ctypes(splits)
        self.splits = sharedctypes.Array(self.splits._type_, self.splits, lock=False)

        # privately expose metadata
        self._labels = labels
        self._unique_label_order = np.unique(labels)
        self._index = index

        # process feature sets if they were provided
        if feature_sets is not None:
            # map features to integer values
            merged_features = np.array(list(reduce(np.union1d, feature_sets.values())))

            # get intersection of features in sets and features in data
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                merged_features = merged_features[np.in1d(
                    merged_features, self._features)]
                if len(merged_features) == 0:
                    raise ValueError('Empty intersection of feature sets and data '
                                     'features')

            # store numerical feature sets, exclude empty sets
            self._numerical_feature_sets = []
            for fset in feature_sets.values():
                fset_idx = np.where(np.in1d(merged_features, fset))[0]
                if fset_idx:
                    self._numerical_feature_sets.append(fset_idx)

            # store the shared indices; this decreases memory usage during sampling
            self._features_in_gene_sets = np.in1d(self._features, merged_features)
            self._feature_set_labels = list(feature_sets.keys())
        else:
            # store empty values to ensure class consistency
            self._numerical_feature_sets = None
            self._features_in_gene_sets = None
            self._feature_set_labels = None

    def _proc_init(self):
        """
        Function run by each process spawned to expose global variables data and splits
        of shared ctypes objects. This is necessary becasue shared ctypes can only be
        passed to processes through inheritance.
        """
        global data
        global splits
        data = self.data
        splits = self.splits

    @staticmethod
    def _draw_sample_with_replacement(normalized_data, n, split_points, column_inds=None):
        """Randomly sample n normalized observations from normalized_data

        :param np.ndarray normalized_data: normalized observations x features matrix. Note
          that the sum of the features for each observation in this data determines the
          total number of times the features are observed.
        :param int n: number of observations to draw from normalized_data
        :param np.ndarray split_points: the locations in normalized_data that if the array
          were split, separate the different classes identified by labels passed to the
          constructor

        :return np.ndarray: integer valued sample: n x features array
        """
        np.random.seed()

        # draw indices from merged array, allocating in a single vector
        idx = np.zeros(n * (len(split_points) + 1), dtype=int)
        esplits = [0] + list(split_points) + [normalized_data.shape[0]]
        for i in np.arange(len(esplits) - 1):
            idx[n*i:n*(i+1)] = np.random.randint(esplits[i], esplits[i+1], n)
        if column_inds is not None:
            sample = normalized_data[idx, :][:, column_inds]
        else:
            sample = normalized_data[idx, :]
        for i in np.arange(sample.shape[0]):  # iteratively to save memory
            p = np.random.sample((1, sample.shape[1]))
            sample[i, :] = np.floor(sample[i, :]) + (sample[i, :] % 1 > p).astype(int)

        return sample

    @staticmethod
    def _draw_sample(normalized_data):
        """Randomly sample each observation from normalized_data a single time, in order

        :param np.ndarray normalized_data: normalized observations x features matrix. Note
          that the sum of the features for each observation in this data determines the
          total number of times the features are observed.

        :return np.ndarray: integer valued sample: n x features array
        """
        np.random.seed()

        # randomly round each data point, generating a sample without replacement
        p = np.random.sample(normalized_data.shape)
        sample = np.floor(normalized_data) + (normalized_data % 1 > p).astype(int)

        return sample

    @staticmethod
    def get_shared_data():
        """helper function to expose a numpy view of the shared data object"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            dview = np.ctypeslib.as_array(data)
        return dview

    @staticmethod
    def get_shared_splits():
        """helper function to expose a numpy view of the shared splits object"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            sview = np.ctypeslib.as_array(splits)
        return sview

    @classmethod
    def mwu_map(cls, n):
        """Mann-Whitney U-test between classes defind by global variables data and splits

        Designed to be mapped to a multiprocessing pool. Draws a sample of size n from
        each class

        :param int n: number of observations to draw with replacement per class sampled
          from data
        :return np.ndarray: n features x 2 array with columns of U-scores and z-scores
        """

        complete_data = cls.get_shared_data()
        array_splits = cls.get_shared_splits()
        assert array_splits.shape[0] == 1  # only have two classes
        xy = cls._draw_sample_with_replacement(complete_data, n, array_splits)
        # calculate U for x
        if xy.ndim == 1:
            xy = xy[:, np.newaxis]
        ranks = rankdata(xy, axis=0)
        del xy  # memory savings
        nt = 2 * n
        U = ranks[:n].sum(0) - n * (n + 1) / 2.

        # get mean value
        mu = (n ** 2) / 2.

        # get smaller u (convention) for reporting only
        u = np.amin([U, n ** 2 - U], axis=0)

        sigsq = np.ones(ranks.shape[1]) * (nt ** 3 - nt) / 12.

        for i in np.arange(len(sigsq)):
            ties = count_tied_groups(ranks[:, i])
            sigsq[i] -= np.sum(v * (k ** 3 - k) for (k, v) in ties.items()) / 12.
        sigsq *= n ** 2 / float(nt * (nt - 1))

        # ignore division by zero warnings; they are properly dealt with by this test.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # note that in the case that sigsq is zero, and mu=U, this line will produce
            # -inf. However, this z-score should be zero, as this is an artifact of
            # continuity correction (subtraction of 1/2)
            z = (U - 1 / 2. - mu) / np.sqrt(sigsq)

        # correct infs
        z[sigsq == 0] = 0

        return np.vstack([u, z]).T

    def mwu_reduce(self, results, alpha=0.05):
        """
        reduction function for Mann-Whitney U-test that processes the results from
        mw_map into a results object

        :param list results: output from mw_map function, a list of np.array objects
          containing U-scores and z-scores.
        :param float alpha: acceptable type-I error rate for BY (negative) FDR correction

        :return pd.DataFrame: contains:
          U: test statistic of M-W U-test
          z_approx: median z-score across iterations
          z_lo: 2.5% confidence boundary for z-score
          z_hi: 97.5% confidence boundary for z-score
          p: p-value corresponding to z_approx
          q: fdr-corrected q-value corresponding to p, across tests in results
        """

        results = np.stack(results)
        ci = confidence_interval(results[:, :, 1])

        results = pd.DataFrame(
            data=np.concatenate([np.median(results, axis=0), ci], axis=1),
            index=self._features,
            columns=['U', 'z_approx', 'z_lo', 'z_hi'])

        # calculate p-values for median z-score
        results['p'] = z_to_p(results['z_approx'])

        # add multiple-testing correction
        results['q'] = multipletests(results['p'], alpha=alpha, method='fdr_by')[1]

        results = results.sort_values('q')
        results.iloc[:, 1:4] = np.round(results.iloc[:, 1:4], 2)
        return results

    @classmethod
    def kw_map(cls, n):
        """Kruskal-wallis ANOVA between classes defind by global variables data and splits

        Designed to be mapped to a multiprocessing pool. Draws a sample of size n from
        each class

        :param int n: number of observations to draw with replacement per class sampled
          from data
        :return np.ndarray: n features x 2 array with columns of H-scores and p-values
        """
        complete_data = cls.get_shared_data()
        ssplits = cls.get_shared_splits()

        sample = cls._draw_sample_with_replacement(complete_data, n, ssplits)

        results = []
        for args in (np.split(sample[:, i], ssplits) for i in np.arange(sample.shape[1])):
            try:
                results.append(_kruskalwallis(*args))
            except ValueError:
                results.append([0, 1.])
        return np.vstack(results)

    def kw_reduce(self, results, alpha=0.05):
        """
        reduction function for Kruskal-Wallis ANOVA that processes the results from
        kw_map into a results object

        :param list results: output from kw_map function, a list of np.array objects
          containing H-scores and z-scores.
        :param float alpha: acceptable type-I error rate for BY (negative) FDR correction

        :return pd.DataFrame: contains:
          H: median test statistic of K-W H-test
          H_lo: 2.5% confidence boundary for H-score
          H_hi: 97.5% confidence boundary for H-score
          p: p-value corresponding to H
          q: fdr-corrected q-value corresponding to p, across tests in results
        """

        results = np.stack(results)  # H, p

        ci = confidence_interval(results[:, :, 0])
        results = pd.DataFrame(
            data=np.concatenate([np.median(results, axis=0), ci], axis=1),
            index=self._features,
            columns=['H', 'p', 'H_lo', 'H_hi'])

        results['q'] = multipletests(results['p'], alpha=alpha, method='fdr_tsbh')[1]
        results = results[['H', 'H_lo', 'H_hi', 'p', 'q']].sort_values('q')
        return results

    @classmethod
    def cluster_map(cls, *args):
        """
        calculate phenograph community assignments across resampled data for each
        of the n observations in global variable data

        :param args: allows passing of arbitrary arguments necessary to evoke
          multiprocessing.Pool.map()
        :return np.ndarray: (n,) cluster assignments
        """
        complete_data = cls.get_shared_data()
        sample = cls._draw_sample(complete_data)
        pca = PCA(n_components=50, svd_solver='randomized')
        reduced = pca.fit_transform(sample)

        # cluster, omitting phenograph outputs
        with suppress_stdout_stderr():
            clusters, *_ = phenograph.cluster(reduced, n_jobs=1)
        return clusters

    # todo consider building this as a sparse matrix (it probably is!)
    def cluster_reduce(self, results):
        """
        carry out consensus clustering across iterations of phenograph run on data,
        running k-means on the confusion matrix generated by these iterations with k fixed
        as the median number of communities selected by phenograph.

        :param list results: list of vectors of cluster labels
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
        return cluster_ids

    @classmethod
    def score_features_map(cls, n, numerical_feature_sets, features_in_gene_sets):
        """score a downsampled group of observations against a set of features

        :param int n: number of observations to draw from each sample
        :param list numerical_feature_sets: list of groups of features, translated to
          numerical indices to support indexing into numpy arrays
        :param np.array features_in_gene_sets:  array of features found in gene sets,
          used to reduce scope of sampling to only relevant features.

        :return np.ndarray: array of scores for n groups defined by different labels
          across p non-empty feature sets
        """
        complete_data = cls.get_shared_data()
        ssplits = cls.get_shared_splits()

        sample = cls._draw_sample_with_replacement(
            complete_data, n, ssplits, features_in_gene_sets)

        results = np.zeros(
            (ssplits.shape[0] + 1, len(numerical_feature_sets)), dtype=float)
        for i, arr in enumerate(np.split(sample, ssplits, axis=0)):
            for j, fset in enumerate(numerical_feature_sets):
                results[i, j] = np.sum(arr[:, fset], axis=1).mean()
        return results

    def score_features_reduce(self, results):
        """
        average over the scores generated by sampling from the groups within the global
        variable data

        :param list results: output of score_features_map, list of numpy arrays containing
          scores
        :return np.ndarray: array of median scores for n groups defined by different
          labels across p non-empty feature sets
        """
        results = np.stack(list(results), axis=2)
        ci = confidence_interval(results, axis=2)
        mu = np.mean(results, axis=2)
        df = pd.Panel(
            np.concatenate([mu[None, :], ci.T], axis=0),
            items=['mean', 'ci_low', 'ci_high'],
            major_axis=self._unique_label_order,
            minor_axis=list(self._feature_set_labels)
        )

        return df

    def run(
            self, n_iter, fmap, freduce, n_processes=None, fmap_kwargs=None,
            freduce_kwargs=None):
        """
        generalized framework to carry out map-reduce sampling over resampled shared data

        :param int n_iter: number of samples to draw
        :param function fmap: function to map over samples of data, must take at least
          one parameter n, the number of observations to draw in each sample
        :param function freduce: function to merge and integrate the results of the
          mapping function. The return value of this function is returned directly by run
        :param int n_processes: number of processes in the multiprocessing Pool. If None,
          this is set to the maximum number of concurrent processes supported by your
          machine
        :param dict fmap_kwargs:
          additional keyword arguments for fmap
        :param dict freduce_kwargs:
          additional keyword arguments for fmap

        :return: result of freduce
        """
        if fmap_kwargs is not None:
            fmap = partial(fmap, **fmap_kwargs)
        if freduce_kwargs is not None:
            freduce = partial(freduce, **freduce_kwargs)

        with closing(Pool(processes=n_processes, initializer=self._proc_init())) as pool:
            results = pool.map(fmap, [self.n_observations_per_sample] * n_iter)
        return freduce(results)
