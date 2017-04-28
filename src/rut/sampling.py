from multiprocessing import Pool, sharedctypes
from itertools import repeat
from contextlib import closing
import numpy as np
import pandas as pd

# must test with openblas enabled and disabled
# see comment #3 on answer #3:
# http://stackoverflow.com/questions/17785275/share-large-read-only-numpy-array-between-multiprocessing-processes
# export OPENBLAS_NUM_THREADS=1

# todo | this module should define a class that acts as a generator to
# todo | draw infinite samples from a normalized data matrix


def round_random():
    pass


def normalize():
    pass


def find_sampling_value():
    pass


def draw_sample(normalized_data, n):
    """Randomly sample n normalized observations from normalized_data
    :param np.ndarray normalized_data: normalized observations x features matrix. Note
      that the sum of the features for each observation in this data determines the
      total number of times the features are observed. Hence, this parameter cannot
      be set in this function, but must be set by normalize()
    :param int n: number of observations to draw from normalized_data
    :param bool return_indices: if True, indices of sampled cells will also be returned

    :return np.ndarray: integer valued sample: n x features array
    """
    np.random.seed()

    idx = np.random.randint(0, normalized_data.shape[0], n)
    sample = normalized_data[idx, :]

    p = np.random.sample(normalized_data.shape)
    sample = np.floor(normalized_data) + (normalized_data % 1 > p).astype(int)

    return sample


class Sampler:

    def __init__(
            self, data, fmap, freduce, n_iter=50, labels=None, sampling_percentile=10,
            upsample=False, is_sorted=False):

        self.fmap = fmap
        self.freduce = freduce
        self.n_iter = n_iter
        self.sampling_percentile = sampling_percentile
        self.upsample = upsample

        # todo add some sanity checks to the input
        if not len(labels) == data.shape[0]:
            raise ValueError(
                'number of labels (%d) does not match number of observations (%d).' %
                (len(labels), data.shape[0]))

        # if user passes sorted data, both labels and data must be contiguous; probably
        # want a safety check here
        self._sorted = is_sorted
        self._is_normalized = False

        # set labels
        if labels is None:  # todo add support for multi-index, warn when used
            if isinstance(data, pd.DataFrame):
                self.labels = np.array(data.index)
            else:
                raise ValueError(
                    'labels must be provided, either as the index of data or as a '
                    'separate array passed using the labels parameter')
        else:
            try:  # should be minimal overhead, I think does nothing if already an array
                self.labels = np.array(labels)
            except:
                raise TypeError('labels must be convertible to a numpy.array type')

        # store indices, if provided
        if isinstance(data, pd.DataFrame):
            self._columns = np.array(data.columns)
        else:
            self._columns = np.arange(data.shape[1])

        # store numeric indices of original array
        self.index = np.arange(data.shape[0])

        # store data
        if isinstance(data, pd.DataFrame):
            self.data = data.values
        elif isinstance(data, np.ndarray):
            self.data = data
        else:
            raise ValueError('data must be a np.ndarray or pd.DataFrame object')

        # default values
        self._numeric_labels = None
        self._numerical_label_order = None
        self._sampling_value = None
        self._observation_sizes = None
        self._results = None
        self._n_observations = None

    @property
    def sampling_value(self):
        return self._sampling_value

    @property
    def shared_data(self):
        return np.ctypeslib.as_array(self.data)

    def __iter__(self):
        """
        access to the underlying sampler; ideally one would just call the function using
        currying.

        :return:
        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        """initialize the pool, run the tests, return the result"""
        raise NotImplementedError

    def _get_numeric_labels(self):
        """
        create and store a numeric translation of labels, as well as their order in the
        final data

        :set np.ndarray self._numerical_label_order: store original category labels in
          their sorted numerical order
        :set np.ndarray self._numerical_labels: store labels translated into integer
          indices
        """
        self._numerical_label_order = np.unique(self.labels)
        if np.issubdtype(self.labels.dtype, np.integer):
            self._numeric_labels = self.labels
        else:
            map_ = dict(zip(self._numerical_label_order, np.arange(self._numerical_label_order.shape[0])))
            self._numeric_labels = np.array([map_[i] for i in self.labels])

    # copies data, 2n memory usage
    def _sort(self):
        """
        sort data by self._numeric_labels. Data and labels are sorted and stored in-place

        :set np.ndarray self.data: store data sorted by numerical labels
        :set np.ndarray self._numeric_labels: store sorted numerical labels
        :set np.ndarray self.labels: store original labels in sorted order
        :set np.ndarray self.index: store original df index in sorted order
        :set bool self._sorted: set sorted flag to True

        :return None:
        """
        if self._sorted:
            return
        idx = np.argsort(self._numeric_labels)
        self.data = self.data[idx, :]  # copies, 2n memory usage, released after function call
        self._numeric_labels = self._numeric_labels[idx]
        self.labels = self.labels[idx]
        self.index = self.index[idx]
        self._sorted = True

    def _identify_splits(self):
        """
        identify break points between classes based on sorted numeric labels

        :set np.ndarray self._splits: break points between classes, used as argument to
          np.split.

        :return None:
        """
        if not self._sorted:
            raise RuntimeError(
                'splits cannot be calculated from unsorted data, run self._sort')
        if self._numeric_labels is None:
            raise RuntimeError(
                'labels must be translated into an integer index before splits can be '
                'calculated. Run self._get_numeric_labels')
        self._splits = np.where(np.diff(self._numeric_labels))[0] + 1

    def _calculate_observation_sizes(self):
        """calculate observation sizes (number of counts in each observation)

        :set np.ndarray self._observation_sizes: store 1d vector of observation sizes
        :return None:
        """
        if not self._sorted:
            raise RuntimeError('data must be sorted before observation sizes can be '
                               'calculated, run self._sort')
        self._observation_sizes = self.data.sum(axis=1)

    def _calculate_n_observations(self):
        """
        calculates the number of observations in the smallest group in self.data. Used to
        determine the size of samples drawn from data.

        :set int self._n_observations: number of observations in smallest group in labels
        :return None:
        """
        self._n_observations = min(d.shape[0] for d in np.split(self.data, self._splits))

    # copies data, 2n memory usage (overwrites data with a sharedmem array)
    def _normalize(self):
        """create a shared array containing data normalized to the correct sampling rate

        :return:
        """
        if self._is_normalized:
            return

        # find the sampling value
        self._get_numeric_labels()
        self._sort()
        self._calculate_observation_sizes()
        self._identify_splits()
        self._sampling_value = min(
            np.percentile(g, self.sampling_percentile) for g in np.split(self._observation_sizes, self._splits))

        # discard cells and labels that are no longer required
        if not self.upsample:
            keep = self._observation_sizes >= self._sampling_value
            self.data = self.data[keep, :]
            self._numeric_labels = self._numeric_labels[keep]
            self.labels = self.labels[keep]
            self.index = self.index[keep]
            self._observation_sizes = self._observation_sizes[keep]

        # normalize data
        self.data = (self.data * self._sampling_value) / self._observation_sizes[:, np.newaxis]
        self._identify_splits()  # update splits

        # convert to shared array
        self.data = np.ctypeslib.as_ctypes(self.data)
        self.data = sharedctypes.Array(self.data._type_, self.data, lock=False)

        self._is_normalized = True

    @staticmethod
    def _process_init(data):
        global shared_data
        shared_data = data

    def run(self, processes):

        # make sure the data is normalized
        self._normalize()

        # mappers should be functions in the form:
        # func(n_observations)

        # note that the original would use self.data, not the numpy wrapper for it.
        with closing(Pool(
                initializer=self._process_init, initargs=(self.data,),
                processes=processes)) as pool:
            results = pool.map(self.fmap, repeat(self._n_observations))

        self._results = self.freduce(results)  # this will be tricky, since a bunch of shit is stored inside the class.
        return results
