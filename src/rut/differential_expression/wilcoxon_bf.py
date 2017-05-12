import numpy as np
from statsmodels.sandbox.stats.multicomp import multipletests
from scipy.stats.mstats import count_tied_groups, rankdata
from rut.differential_expression import differential_expression
import rut.misc
import pandas as pd


class WilcoxonBF(differential_expression.DifferentialExpression):

    def __init__(self, *args, **kwargs):
        """

        # todo enumerate actual arguments from super class too
        # todo add plotting functions
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        if self._labels is None:
            raise ValueError('Labels are required for WilcoxonBF Test')
        elif np.unique(self._labels).shape[0] != 2:
            raise ValueError(
                'Labels must contain only two categories for WilcoxonBF Testing. '
                'Please use KruskalWallis for poly-sample comparisons')

    # todo determine if statistic's t-approximation is necessary to implement (large n)
    @classmethod
    def _map(cls, n):
        """
        Wilcoxon Behrens-Fisher test between classes defind by global variables data
        and splits

        Designed to be mapped to a multiprocessing pool. Draws a sample of size n from
        each class

        :param int n: number of observations to draw with replacement per class sampled
          from data
        :return np.ndarray: n features x 2 array with columns of test statistics and
          p-values
        """

        def empirical_variance(rw, rb, mu_r, n):
            """

            :param np.ndarray rw: n samples x g genes within-sample ranks
            :param np.ndarray rb: n samples x g genes between-sample ranks
            :param np.ndarray mu_r: g genes mean ranks
            :param int n: number of samples
            :return:
            """
            s2 = (1 / (n - 1)) * np.sum((rb - rw - mu_r + ((n + 1) / 2)) ** 2, axis=0)
            return s2 / n ** 2

        complete_data = cls.get_shared_data()
        array_splits = cls.get_shared_splits()
        assert array_splits.shape[0] == 1  # only have two classes
        xy = cls._draw_sample_with_replacement(complete_data, n, array_splits)
        # calculate U for x
        if xy.ndim == 1:
            xy = xy[:, np.newaxis]
        ranks = rankdata(xy, axis=0)  # 2n x g
        x_ranks = rankdata(xy[:n, :])  # n x g
        y_ranks = rankdata(xy[n:, :])  # n x g
        del xy  # memory savings

        # calculate variance
        mean_rank = np.mean(ranks, axis=0)  # todo this can be computed faster, and is an int not an array
        s2_x = empirical_variance(x_ranks, ranks[:n, :], mean_rank, n)
        s2_y = empirical_variance(y_ranks, ranks[n:, :], mean_rank, n)
        sigma = 2*n * ((s2_x / n) + (s2_y / n))

        # calculate statistic
        w = (1 / np.sqrt(2 * n) *
             (np.mean(ranks[n:, :], axis=0) - np.mean(ranks[:n, :], axis=0)) / sigma)

        return w

    def _reduce(self, results, alpha=0.05):
        """
        reduction function for Wilcoxon Behrens-Fisher test that processes the results
        from self._map into a results object

        :param list results: output from mw_map function, a list of np.array objects
          containing z-scores
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
        ci = rut.misc.confidence_interval(results)
        p = rut.misc.z_to_p(results.sum(axis=0) / np.sqrt(results.shape[0]))  # higher power fisher method
        results = pd.DataFrame(
            data=np.concatenate([np.median(results, axis=0)[:, None], ci, p[:, None]], axis=1),
            index=self._features,
            columns=['W', 'W_low', 'W_high', 'p'])

        # calculate p-values for median z-score
        # results['p'] = rut.misc.z_to_p(results['W'])  # median z

        # add multiple-testing correction
        results['q'] = multipletests(results['p'], alpha=alpha, method='fdr_by')[1]

        results = results.sort_values('q')
        results.iloc[:, 1:4] = np.round(results.iloc[:, 1:4], 2)
        return results

    def fit(self, n_iter=50, n_processes=None, alpha=0.05):
        """
        Carry out a Behrens-Fisher test

        :return:
        """
        self.result_ = self.run(
            n_iter=n_iter,
            n_processes=n_processes,
            fmap=self._map,
            freduce=self._reduce,
            freduce_kwargs=dict(alpha=alpha)
        )

        return self.result_
