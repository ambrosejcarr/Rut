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

# np.seterr(all='warn')
# warnings.filterwarnings("error", category=DeprecationWarning)
# warnings.filterwarnings("error", category=FutureWarning)
# must test with openblas enabled and disabled
# see comment #3 on answer #3:
# http://stackoverflow.com/questions/17785275/share-large-read-only-numpy-array-between-multiprocessing-processes
# export OPENBLAS_NUM_THREADS=1

# bug in python produces a runtime warning; grab this with warnings and eliminate it
# there are maybe some ways of working around this, see:
# http://stackoverflow.com/questions/4964101/pep-3118-warning-when-using-ctypes-array-as-numpy-array


class Sampled:

    def __init__(
            self, data, labels, sampling_percentile=10, feature_sets=None, upsample=False,
            is_sorted=False):
        """

        :param pd.DataFrame | np.array data:
        :param np.ndarray labels:
        :param sampling_percentile:
        :param feature_sets: optional parameter, pre-processing done in init
        :param upsample:
        :param is_sorted:
        """

        assert len(labels) == data.shape[0], '%d != %d' % (len(labels), data.shape[0])
        # assume DataFrame for now, fix later
        if not isinstance(data, pd.DataFrame):
            raise TypeError('use dataframe')
        else:
            index = data.index
            self._features = data.columns
            data = data.values

        # make labels numeric; does nothing if labels are already numeric
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
        sampling_value = min(
            np.percentile(g, sampling_percentile)
            for g in np.split(observation_sizes, splits))

        # throw out cells below sampling threshold unless upsampling is requested
        if not upsample:
            keep = observation_sizes >= sampling_value
            data = data[keep, :]
            numeric_labels = numeric_labels[keep]
            labels = labels[keep]
            index = index[keep]
            observation_sizes = observation_sizes[keep]
            splits = np.where(np.diff(numeric_labels))[0] + 1
        self.n = min(g.shape[0] for g in np.split(observation_sizes, splits))

        # normalize data
        data = (data * sampling_value) / observation_sizes[:, np.newaxis]

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

            # toss features not in data
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                merged_features = merged_features[np.in1d(
                    merged_features, self._features)]
                if len(merged_features) == 0:
                    raise ValueError('Empty intersection of feature sets and data '
                                     'features')
            self._numerical_feature_sets = [
                np.where(np.in1d(merged_features, fset))[0] for fset in
                feature_sets.values()]

            self._features_in_gene_sets = np.in1d(self._features, merged_features)
            self._feature_set_labels = list(feature_sets.keys())
        else:
            self._numerical_feature_sets = None
            self._features_in_gene_sets = None
            self._feature_set_labels = None

    def _proc_init(self):
        global data
        global splits
        data = self.data
        splits = self.splits

    @staticmethod
    def _draw_single_sample_with_replacement(normalized_data, n, splits, column_inds=None):
        """Randomly sample n normalized observations from normalized_data
        :param np.ndarray normalized_data: normalized observations x features matrix. Note
          that the sum of the features for each observation in this data determines the
          total number of times the features are observed. Hence, this parameter cannot
          be set in this function, but must be set by normalize()
        :param int n: number of observations to draw from normalized_data

        :return np.ndarray: integer valued sample: n x features array
        """
        np.random.seed()

        # draw indices from merged array, allocating in a single vector
        idx = np.zeros(n * (len(splits) + 1), dtype=int)
        esplits = [0] + list(splits) + [normalized_data.shape[0]]
        for i in np.arange(len(esplits) - 1):
            idx[n*i:n*(i+1)] = np.random.randint(esplits[i], esplits[i+1], n)
        print(idx)
        if column_inds is not None:
            sample = normalized_data[idx, :][:, column_inds]
        else:
            sample = normalized_data[idx, :]
        for i in np.arange(sample.shape[0]):  # iteratively to save memory
            p = np.random.sample((1, sample.shape[1]))
            sample[i, :] = np.floor(sample[i, :]) + (sample[i, :] % 1 > p).astype(int)

        return sample

    @staticmethod
    def _draw_sample(normalized_data, n):
        """Randomly sample n normalized observations from normalized_data
        :param np.ndarray normalized_data: normalized observations x features matrix. Note
          that the sum of the features for each observation in this data determines the
          total number of times the features are observed. Hence, this parameter cannot
          be set in this function, but must be set by normalize()
        :param int n: number of observations to draw from normalized_data

        :return np.ndarray: integer valued sample: n x features array
        """
        np.random.seed()

        # idx = np.random.randint(0, normalized_data.shape[0], n)
        # sample = normalized_data[idx, :]

        p = np.random.sample(normalized_data.shape)
        sample = np.floor(normalized_data) + (normalized_data % 1 > p).astype(int)

        return sample

    @staticmethod
    def get_group_data():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            dview = np.ctypeslib.as_array(data)
            sview = np.ctypeslib.as_array(splits)
        return np.split(dview, sview)

    @staticmethod
    def get_ungrouped_data():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            dview = np.ctypeslib.as_array(data)
        return dview

    @staticmethod
    def get_splits():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            sview = np.ctypeslib.as_array(splits)
        return sview

    @classmethod
    def mwu_map(cls, n):

        complete_data = cls.get_ungrouped_data()
        array_splits = cls.get_splits()
        assert array_splits.shape[0] == 1  # only have two classes
        xy = cls._draw_single_sample_with_replacement(complete_data, n, array_splits)
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

        prob = z_to_p(z)
        return np.vstack([u, z, prob]).T

    def mwu_reduce(self, results, alpha=0.05):

        results = np.stack(results)
        ci = confidence_interval(results[:, :, 1])

        results = pd.DataFrame(
            data=np.concatenate([np.median(results, axis=0), ci], axis=1),
            index=self._features,
            columns=['U', 'z_approx', 'p', 'z_lo', 'z_hi'])

        # add multiple-testing correction
        results['q'] = multipletests(results['p'], alpha=alpha, method='fdr_tsbh')[1]

        results = results[['U', 'z_approx', 'z_lo', 'z_hi', 'p', 'q']].sort_values('q')
        results.iloc[:, 1:4] = np.round(results.iloc[:, 1:4], 2)
        return results

    @classmethod
    def kw_map(cls, n):
        group_data = cls.get_group_data()
        samples = [cls._draw_sample_with_replacement(d, n) for d in group_data]

        results = []
        for i in np.arange(samples[0].shape[1]):
            args = [d[:, i] for d in samples]
            try:
                results.append(_kruskalwallis(*args))
            except ValueError:
                results.append([0, 1.])
        return results

    def kw_reduce(self, results, alpha=0.05):
        # todo | this function currently lacks sorting of results, which is not
        # todo | consistent with MWU
        results = np.stack(results)  # H, p

        ci = confidence_interval(results[:, :, 0])  # around H
        results = pd.DataFrame(
            data=np.concatenate([np.median(results, axis=0), ci], axis=1),
            index=self._features,
            columns=['H', 'p', 'H_lo', 'H_hi'])

        results['q'] = multipletests(results['p'], alpha=alpha, method='fdr_tsbh')[1]
        results = results[['H', 'H_lo', 'H_hi', 'p', 'q']]
        return results

    @classmethod
    def cluster_map(cls, n):
        sample = cls.get_ungrouped_data()
        pca = PCA(n_components=50, svd_solver='randomized')
        reduced = pca.fit_transform(sample)

        # cluster, omitting phenograph outputs
        with suppress_stdout_stderr():
            clusters, *_ = phenograph.cluster(reduced, n_jobs=1)
        return clusters

    def cluster_reduce(self, results):
        n = results[0].shape[0]
        integrated = np.zeros((n, n), dtype=np.uint8)
        for i in np.arange(len(results)):
            labs = results[i]
            for j in np.arange(integrated.shape[0]):
                increment = labs == labs[j]
                integrated[j, :] += increment.astype(np.uint8)

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
        group_data = cls.get_group_data()
        samples = [
            cls._draw_sample_with_replacement(d, n, features_in_gene_sets)
            for d in group_data]
        results = np.zeros((len(samples), len(numerical_feature_sets)), dtype=float)
        for i, arr in enumerate(samples):
            for j, fset in enumerate(numerical_feature_sets):
                results[i, j] = np.sum(arr[:, fset], axis=1).mean()
        return results

    def score_features_reduce(self, results):
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

        :return:
        """
        if fmap_kwargs is not None:
            fmap = partial(fmap, **fmap_kwargs)
        if freduce_kwargs is not None:
            freduce = partial(freduce, **freduce_kwargs)

        with closing(Pool(processes=n_processes, initializer=self._proc_init())) as pool:
            results = pool.map(fmap, [self.n] * n_iter)
        return freduce(results)
