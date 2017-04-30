from contextlib import closing
from multiprocessing import Pool, sharedctypes
import numpy as np
import pandas as pd
from scipy.stats.mstats import count_tied_groups, rankdata
import warnings
from rut.stats import z_to_p, confidence_interval
from statsmodels.sandbox.stats.multicomp import multipletests

# must test with openblas enabled and disabled
# see comment #3 on answer #3:
# http://stackoverflow.com/questions/17785275/share-large-read-only-numpy-array-between-multiprocessing-processes
# export OPENBLAS_NUM_THREADS=1

# bug in python produces a runtime warning; grab this with warnings and eliminate it
# there are maybe some ways of working around this, see:
# http://stackoverflow.com/questions/4964101/pep-3118-warning-when-using-ctypes-array-as-numpy-array


class Sampled:

    def __init__(
            self, data, labels, sampling_percentile=10, upsample=False, is_sorted=False):
        """

        :param pd.DataFrame | np.array data:
        :param n_iter:
        :param np.ndarray labels:
        :param sampling_percentile:
        :param upsample:
        :param is_sorted:
        """
        # assume DataFrame for now, fix later
        if not isinstance(data, pd.DataFrame):
            raise TypeError('use dataframe')
        else:
            index = data.index
            self.features = data.columns
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

        self._labels = labels
        self._index = index

    def _proc_init(self):
        global data
        global splits
        data = self.data
        splits = self.splits

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

        idx = np.random.randint(0, normalized_data.shape[0], n)
        sample = normalized_data[idx, :]

        p = np.random.sample(sample.shape)
        sample = np.floor(sample) + (sample % 1 > p).astype(int)

        return sample

    @staticmethod
    def datasum(n):
        return np.ctypeslib.as_array(data).sum()

    @classmethod
    def mwu_test(cls, n):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            dview = np.ctypeslib.as_array(data)
            sview = np.ctypeslib.as_array(splits)
        group_data = np.split(dview, sview)
        assert len(group_data) == 2
        x, y = (cls._draw_sample(d, n) for d in group_data)

        # calculate U for n1
        if x.ndim == 1 and y.ndim == 1:
            x, y = x[:, np.newaxis], y[:, np.newaxis]
        ranks = rankdata(np.concatenate([x, y]), axis=0)
        nx, ny = x.shape[0], y.shape[0]
        nt = nx + ny
        U = ranks[:nx].sum(0) - nx * (nx + 1) / 2.

        # get mean value
        mu = (nx * ny) / 2.

        # get smaller u (convention) for reporting only
        u = np.amin([U, nx * ny - U], axis=0)

        sigsq = np.ones(ranks.shape[1]) * (nt ** 3 - nt) / 12.

        for i in np.arange(len(sigsq)):
            ties = count_tied_groups(ranks[:, i])
            sigsq[i] -= np.sum(v * (k ** 3 - k) for (k, v) in ties.items()) / 12.
        sigsq *= nx * ny / float(nt * (nt - 1))

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

    def mwu_reduce(self, results, alpha=0.5):

        results = np.stack(results)
        ci = confidence_interval(results[:, :, 1])

        results = pd.DataFrame(
            data=np.concatenate([np.median(results, axis=0), ci], axis=1),
            index=self.features,
            columns=['U', 'z_approx', 'p', 'z_lo', 'z_hi'])

        # add multiple-testing correction
        results['q'] = multipletests(results['p'], alpha=alpha, method='fdr_tsbh')[1]

        results = results[['U', 'z_approx', 'z_lo', 'z_hi', 'p', 'q']].sort_values('q')
        results.iloc[:, 1:4] = np.round(results.iloc[:, 1:4], 2)
        return results

    def run(self, n_iter, fmap, freduce, n_processes=None):
        """

        :return:
        """
        with closing(Pool(processes=n_processes, initializer=self._proc_init())) as pool:
            results = pool.map(fmap, [self.n] * n_iter)
        return freduce(results)
