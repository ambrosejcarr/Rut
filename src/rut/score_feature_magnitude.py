from functools import reduce
import warnings
import numpy as np
import pandas as pd
from rut import sample, misc


class ScoreFeatureMagnitudes(sample.Sampled):

    def __init__(self, data, labels, feature_groups, *args, **kwargs):

        # check inputs
        if not isinstance(feature_groups, dict):
            raise TypeError(
                'feature_groups must be a dictionary of iterables containing features, '
                'not %s' % repr(type(feature_groups)))

        # store numerical feature sets, exclude empty sets
        self._numerical_feature_sets = []
        self._feature_set_labels = []
        for k, fset in feature_groups.items():
            fset = np.intersect1d(fset, data.columns)
            fset_idx = np.where(np.in1d(data.columns, fset))[0]
            if fset_idx.shape[0] != 0:
                self._numerical_feature_sets.append(fset_idx)
                self._feature_set_labels.append(k)

        super().__init__(data, labels, *args, **kwargs)

    @classmethod
    def score_features_map(cls, n, numerical_feature_sets):
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

        sample_ = cls._draw_sample_with_replacement(complete_data, n, ssplits)

        results = np.zeros(
            (ssplits.shape[0] + 1, len(numerical_feature_sets)), dtype=float)
        for i, arr in enumerate(np.split(sample_, ssplits, axis=0)):
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
        ci = misc.confidence_interval(results, axis=2)
        mu = np.mean(results, axis=2)
        df = pd.Panel(
            np.concatenate([mu[None, :], ci.T], axis=0),
            items=['mean', 'ci_low', 'ci_high'],
            major_axis=self._unique_label_order,
            minor_axis=list(self._feature_set_labels)
        )

        return df

    # todo document me
    def fit(self, n_iter=50, n_processes=None):
        """

        :param n_iter:
        :param n_processes:
        :return:
        """
        return self.run(
            n_iter=n_iter,
            fmap=self.score_features_map,
            freduce=self.score_features_reduce,
            n_processes=n_processes,
            fmap_kwargs=dict(numerical_feature_sets=self._numerical_feature_sets)
        )

