from contextlib import closing
from multiprocessing import Pool, sharedctypes
import numpy as np
import pandas as pd
import abc
import warnings
from functools import partial

# todo test with openblas enabled and disabled
# see comment #3 on answer #3:
# http://stackoverflow.com/questions/17785275/share-large-read-only-numpy-array-\
# between-multiprocessing-processes
# export OPENBLAS_NUM_THREADS=1


class Sampled(abc.ABC):
    """
    The Abstract Base Class for all methods using this sampling approach to query
    expression matrices.

    This class defines concrete methods that implement a sampling approach to drawing
    fast samples with or without replacement using a shared memory numpy array accessible
    across a multiprocessing pool.

    It also defines three abstract methods which must be implemented by inheriting
    classes. These methods define a map-reduce framework which together define a
    statistical bootstrap wherein samples are generated and utilized to produce a test
    statistic (map) which is then summarized (reduce) and wrapped for easy use (fit):

    :method map: The map method defines a single iteration of the sampling procedure and
      is the method called by the multiprocessing pool to execute each iteration.
    :method reduce: The reduce method integrates information from each of the iterations
      of sampling
    :method fit: Public method called by the user to execute the
    """

    def __init__(self, data, labels=None, is_sorted=False, max_obs_per_sample=500):
        """

        :param pd.DataFrame | np.array data: m observations x p features array or
          dataframe
        :param np.ndarray labels: m x 1 condition labels that separate cells into groups
          of comparison
        :param bool is_sorted: if True, the dataframe index is assumed to be pre-sorted
          based on the provided labels. No additional sorting is done of data or labels
        :param int max_obs_per_sample: hard ceiling on the number of observations to
          take for each sample. Useful for constraining memory usage for very large
          groups. More than 1000 observations in the smallest group can use > 2gb memory
          per process. Default = 500.
        """

        # check input types
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

        # get number of draws per sample
        observation_sizes = data.sum(axis=1)
        self.size_of_samples = np.min(observation_sizes)

        # get number of samples to draw
        if labels is not None:
            if len(labels) != data.shape[0]:
                raise ValueError('number of labels %d != number of observations %d'
                                 % (len(labels), data.shape[0]))

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
            splits_ = np.where(np.diff(numeric_labels))[0] + 1

            self.n_samples_to_draw = min(
                g.shape[0] for g in np.split(observation_sizes, splits_))
            if max_obs_per_sample is not None:
                self.n_samples_to_draw = min(max_obs_per_sample, self.n_samples_to_draw)

            # expose splits
            self.splits = np.ctypeslib.as_ctypes(splits_)
            self.splits = sharedctypes.Array(self.splits._type_, self.splits, lock=False)

        else:
            self.n_samples_to_draw = min(observation_sizes)
            self.splits = None

        # normalize data
        data = (data * self.size_of_samples) / observation_sizes[:, np.newaxis]

        # convert to shared array & expose data
        self.data = np.ctypeslib.as_ctypes(data)
        self.data = sharedctypes.Array(self.data._type_, self.data, lock=False)

        # privately expose metadata
        self._labels = labels
        self._unique_label_order = np.unique(labels)
        self._index = index

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
            results = pool.map(fmap, [self.n_samples_to_draw] * n_iter)
        return freduce(results)

    @classmethod
    @abc.abstractmethod
    def map(cls, n):
        """The map method should define how a single iteration of the bootstrap is
        carried out. It should typically call _draw_sample or
        _draw_sample_with_replacement and use the sample to calculate a statistic defined
        by the inheriting class.

        :param int n: the number of observations to sample for this iteration
        :return np.ndarray: numpy array of length p (features). typically contains test
          statistics and may optionally also contain p-values.
        """
        return

    @abc.abstractmethod
    def reduce(self, results, alpha):
        """The reduce method should integrate the information obtained by each of the
        iterations of map. If map generates a distribution of test statistics, reduce
        might determine the central value of the distribution and calculate a confidence
        interval.

        :param list results: list of outputs generated by calling map n_iter times using
          the multiprocessing.pool method
        :param float alpha: acceptable type-I error rate for BY (negative) FDR correction.
        :return pd.DataFrame: containing summarized results for each tested feature
        """
        return

    @abc.abstractmethod
    def fit(self, n_iter, n_processes, alpha):
        """Public method that carries out a bootstrapped test using n_processes to run
        n_iter samples. Resulting data are corrected using an FDR approach, fixing
        allowable type-I error at alpha.

        :param int n_iter: number of sampling iterations to run
        :param int n_processes: number of processes to use in the pool (default = number
          available to runtime environment)
        :param float alpha: allowable type-I error
        :return:
        """
        return