import numpy as np
import pandas as pd
from statsmodels.sandbox.stats.multicomp import multipletests
from scipy.stats.mstats import kruskalwallis as _kruskalwallis
from rut import misc
from rut.differential_expression import differential_expression
from rut.sample import Sampled


class KruskalWallis(Sampled, differential_expression.DifferentialExpression):
    """
    Kruskal-Wallis H-test (Non-parametric ANOVA) over down-sampled data from two or
    more groups of cells.
    """

    def __init__(self, *args, **kwargs):
        """

        :param pd.DataFrame | np.array data: m observations x p features array or
          dataframe
        :param np.ndarray labels: p x 1 condition labels that separate cells into units of
          comparison
        :param bool is_sorted: if True, no sorting is done of data or labels
        :param int max_obs_per_sample: hard ceiling on the number of observations to
          take for each sample. Useful for constraining memory usage

        """

        super().__init__(*args, **kwargs)
        if self._labels is None:
            raise ValueError('Labels are required for Differential Expression Testing.')

    @classmethod
    def map(cls, n):
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

    def reduce(self, results, alpha=0.05):
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

        ci = misc.confidence_interval(results[:, :, 0])
        results = pd.DataFrame(
            data=np.concatenate([np.median(results, axis=0), ci], axis=1),
            index=self._features,
            columns=['H', 'p', 'H_lo', 'H_hi'])

        results['q'] = multipletests(results['p'], alpha=alpha, method='fdr_tsbh')[1]
        results = results[['H', 'H_lo', 'H_hi', 'p', 'q']].sort_values('q')
        return results

    def fit(self, n_iter=50, n_processes=None, alpha=0.05):
        """
        Carry out a Kruskal-Wallis ANOVA across the groups

        :return pd.DataFrame: contains:
          H: median test statistic of K-W H-test
          H_lo: 2.5% confidence boundary for H-score
          H_hi: 97.5% confidence boundary for H-score
          p: p-value corresponding to H
          q: fdr-corrected q-value corresponding to p, across tests in results
        """
        self.result_ = self.run(
            n_iter=n_iter,
            n_processes=n_processes,
            fmap=self.map,
            freduce=self.reduce,
            freduce_kwargs=dict(alpha=alpha)
        )

        return self.result_
