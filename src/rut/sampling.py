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


def round_random(sample):
    """round a sample probabilistically; this is the most computationally expensive function in this module"""
    p = np.random.sample(sample.shape)
    return np.floor(sample) + (sample % 1 > p).astype(int)


class Sampler:

    def __init__(self, data, fmap, freduce, n_iter=50, labels=None, sampling_percentile=10, upsample=False,
                 is_sorted=False):

        self.fmap = fmap
        self.freduce = freduce
        self.n_iter = n_iter
        self.sampling_percentile = sampling_percentile
        self.upsample = upsample

        # todo add some sanity checks to the input
        if not len(labels) == data.shape[0]:
            raise ValueError('number of labels (%d) does not match number of observations (%d).' %
                             (len(labels), data.shape[0]))

        # if user passes sorted data, both labels and data must be contiguous; probably want a safety check here
        self._sorted = is_sorted
        self._normalized = False

        # set labels
        if labels is None:
            if isinstance(data, pd.DataFrame):
                self.labels = np.array(data.index)  # todo add support for multi-index, warn when used
            else:
                raise ValueError('labels must be provided, either as the index of data or as a separate array passed '
                                 'using the labels parameter')
        else:
            try:
                self.labels = np.array(labels)  # should be minimal overhead, I think does nothing if already an array
            except:
                raise TypeError('labels must be convertible to a numpy.array type')

        # store indices, if provided
        if isinstance(data, pd.DataFrame):
            self._columns = np.array(data.columns)
        else:
            self._columns = np.arange(data.columns)

        # store numeric indices of original array
        self.index = np.arange(data.shape[0])

        # store data
        self.data = data.values

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
        """access to the underlying sampler; ideally one would just call the function using currying.

        :return:
        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        """initialize the pool, run the tests, return the result"""
        raise NotImplementedError

    def _get_numeric_labels(self):
        """create and store a numeric translation of labels, as well as their order in the final data"""
        self._numerical_label_order = np.unique(self.labels)
        if np.issubdtype(self.labels.dtype, np.integer):
            self._numeric_labels = self.labels
        else:
            map_ = dict(zip(self._numerical_label_order, np.arange(self._numerical_label_order.shape[0])))
            self._numeric_labels = np.array([map_[i] for i in self.labels])

    # copies data, 2n memory usage
    def _sort(self):
        """order data by labels into contiguous groups, data and numeric labels are updated in-place"""
        if self._sorted:
            return
        idx = np.argsort(self._numeric_labels)
        self.data = self.data[idx, :]  # copies, 2n memory usage, released after function call
        self._numeric_labels = self._numeric_labels[idx]
        self.labels = self.labels[idx]
        self.index = self.index[idx]
        self._sorted = True

    def _identify_splits(self):
        """identify break points between classes"""
        self._splits = np.where(np.diff(self._numeric_labels))[0] + 1

    def _calculate_observation_sizes(self):
        if not self._sorted:
            raise RuntimeError('data must be sorted before observation sizes can be calculated')
        self._observation_sizes = self.data.sum(axis=1)

    def _calculate_n_observations(self):
        self._n_observations = min(d.shape[0] for d in np.split(self.data, self._splits))

    # copies data, 2n memory usage (overwrites data with a sharedmem array)
    def _normalize(self):
        """create a shared array containing data normalized to the correct sampling rate

        :return:
        """

        # find the sampling value
        self._get_numeric_labels()
        self._sort()
        self._calculate_observation_sizes()
        self._sampling_value = min(
            np.percentile(g, self.sampling_percentile) for g in np.split(self._observation_sizes, self._splits))

        # discard cells and labels that are no longer required
        if not self.upsample:
            keep = self._observation_sizes >= self._sampling_value
            data = self.data[keep, :]
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

        self.is_normalized = True

    @staticmethod
    def _process_init(data):
        global shared_data
        shared_data = data

    def _run(self):

        # make sure the data is normalized

        # mappers should be functions in the form:
        # func(n_observations)

        # note that the original would use self.data, not the numpy wrapper for it.
        with closing(Pool(initializer=self._process_init, initargs=(self.data,))) as pool:
            results = pool.map(self.fmap, repeat(self._n_observations))

        self._results = self.freduce(results)  # this will be tricky, since a bunch of shit is stored inside the class.
        return results


def find_sampling_value(group_data, percentile):
    """identify the sampling value

    :param list group_data: among each group, the row sum at percentile is
      evaluated, and the minimum value across the groups is the sampling value
    :param int percentile: the percentile to select for the sampling value

    :return int: the minimum value at percentile across the groups in group_data
    """
    return min(np.percentile(g.sum(axis=1), percentile) for g in group_data)


def normalize(data, downsample_value, upsample=False, labels=None):
    """Normalize data such that each observation sums to downsample_value

    This function has an important downstream interaction with _draw_sample. In draw
    sample, non-integer values are rounded probabilistically, such that integer outputs
    are produced this has the effect of producing observations with very similar
    distributions to their down-sampled observations, but with more uncertainty
    associated with observations that have sparser sampling

    :param np.ndarray data: array containing observations x features
    :param int downsample_value: value to normalize observation counts to.
    :param bool upsample: if False, all observations with size < downsample_value are
      excluded. If True, those observations are upsampled to downsample_value.
    :param np.ndarray labels: array or series of labels, useful to track which cells
      are being discarded by exclusion of observations with sums below the downsampling
      value
    :return np.ndarray: array containing observations x samples where rows sum to
      downsample_value
    """
    obs_size = data.sum(axis=1)
    if not upsample:
        keep = obs_size >= downsample_value
        data = data[keep, :]
        if labels is not None:
            labels = labels[keep]
    norm = (data * downsample_value) / data.sum(axis=1)[:, np.newaxis]
    if labels is not None:
        return norm, labels
    else:
        return norm


def round_random(sample):
    """round a sample probabilistically"""
    p = np.random.sample(sample.shape)
    return np.floor(sample) + (sample % 1 > p).astype(int)


def draw_sample(normalized_data, n, return_indices=False):
    """Randomly sample n normalized observations from normalized_data
    :param np.ndarray normalized_data: normalized observations x features matrix. Note
      that the sum of the features for each observation in this data determines the
      total number of times the features are observed. Hence, this parameter cannot
      be set in this function, but must be set by normalize()
    :param int n: number of observations to draw from normalized_data
    :param bool return_indices: if True, indices of sampled cells will also be returned

    :return np.ndarray: n x features array
    """
    np.random.seed()
    # select cells
    idx = np.random.randint(0, normalized_data.shape[0], n)
    sample = normalized_data[idx, :]

    sample = round_random(sample)
    if return_indices:
        return sample, idx
    else:
        return sample
