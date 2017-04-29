from contextlib import closing
from multiprocessing import Pool, sharedctypes
import numpy as np
import pandas as pd

# must test with openblas enabled and disabled
# see comment #3 on answer #3:
# http://stackoverflow.com/questions/17785275/share-large-read-only-numpy-array-between-multiprocessing-processes
# export OPENBLAS_NUM_THREADS=1


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
            features = data.columns
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

    def _proc_init(self):
        global data
        data = self.data

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

    def run(self, n_iter, fmap, freduce, n_processes=None):
        """

        :return:
        """
        with closing(Pool(processes=n_processes, initializer=self._proc_init())) as pool:
            results = pool.map(fmap, [self.n] * n_iter)
        return freduce(results)
