import numpy as np
from statsmodels.sandbox.stats.multicomp import multipletests
from scipy.stats.mstats import count_tied_groups, rankdata
from rut.differential_expression import differential_expression
import rut.misc
import pandas as pd
from rut.sample import Sampled


class WilcoxonBF(Sampled, differential_expression.DifferentialExpression):
    """
    Wilcoxon BF-test over down-sampled data from two groups of cells.
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
            raise ValueError('Labels are required for WilcoxonBF Test')
        elif np.unique(self._labels).shape[0] != 2:
            raise ValueError(
                'Labels must contain only two categories for WilcoxonBF Testing. '
                'Please use KruskalWallis for poly-sample comparisons')

    # todo determine if statistic's t-approximation is necessary to implement (large n)
    @classmethod
    def map(cls, n):
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

        def empirical_variance(within_rank, total_rank, mean_total_rank, n, N):
            """calculate the empirical variance of the sample

            :param np.ndarray within_rank: n samples x g genes within-sample ranks
            :param np.ndarray total_rank: n samples x g genes between-sample ranks
            :param np.ndarray mean_total_rank: g genes mean ranks
            :param int n: number of samples in this group
            :param int N: number of samples in all groups
            :return:
            """
            adjustment_factor = (n + 1) / 2
            ranks = total_rank - within_rank - mean_total_rank + adjustment_factor
            z = 1 / (n - 1)
            s2 = z * np.sum(ranks ** 2, axis=0)
            sigma2 = s2 / (N - n) ** 2
            return sigma2

        def wbf(xy, n_x, n_y):
            """calculate the wbf test statistic

            :param np.ndarray xy: concatenated data for x and y
            :param int n_x: number of observations of x
            :param int n_y: number of observations of y
            :return:
            """
            x_ranks = rankdata(xy[:n_x, :], axis=0)
            y_ranks = rankdata(xy[n_x:, :], axis=0)
            N = n_x + n_y
            xy_ranks = rankdata(xy, axis=0)
            x_mean_rank = np.mean(xy_ranks[:n_x, :], axis=0)
            y_mean_rank = np.mean(xy_ranks[n_x:, :], axis=0)
            sigma2_x = empirical_variance(x_ranks, xy_ranks[:n_x, :], x_mean_rank, n_x, N)
            sigma2_y = empirical_variance(y_ranks, xy_ranks[n_x:, :], y_mean_rank, n_y, N)
            sigma_pool = np.sqrt(N * (sigma2_x / n_x + sigma2_y / n_y))
            W = 1 / np.sqrt(N) * ((y_mean_rank - x_mean_rank) / sigma_pool)
            return W

        complete_data = cls.get_shared_data()
        array_splits = cls.get_shared_splits()
        assert array_splits.shape[0] == 1  # only have two classes
        xy = cls._draw_sample_with_replacement(complete_data, n, array_splits)
        # calculate U for x
        if xy.ndim == 1:
            xy = xy[:, np.newaxis]

        # calculate test statistic
        return wbf(xy, n, n)  # n_x and n_y are both equal to the sampling size

    def reduce(self, results, alpha=0.05):
        """
        reduction function for Wilcoxon Behrens-Fisher test that processes the results
        from self._map into a results object

        :param list results: output from mw_map function, a list of np.array objects
          containing z-scores
        :param float alpha: acceptable type-I error rate for BY (negative) FDR correction

        :return pd.DataFrame: contains:
          W: test statistic of WBF test
          W_low: 2.5% confidence boundary for W
          W_high: 97.5% confidence boundary for W
          p: p-value corresponding to W
          q: fdr-corrected q-value corresponding to p, across tests in results
        """

        results = np.stack(results)
        results[np.isnan(results)] = 0  # todo fix this
        ci = rut.misc.confidence_interval(results)
        results = pd.DataFrame(
            data=np.concatenate([np.mean(results, axis=0)[:, None], ci], axis=1),
            index=self._features,
            columns=['W', 'W_low', 'W_high'])
        results['p'] = rut.misc.z_to_p(results['W'])  # mean z

        # add multiple-testing correction
        results['q'] = multipletests(results['p'], alpha=alpha, method='fdr_by')[1]

        results = results.sort_values('q')
        results.iloc[:, 1:4] = np.round(results.iloc[:, 1:4], 2)
        return results

    def fit(self, n_iter=50, n_processes=None, alpha=0.05):
        """
        Carry out a Behrens-Fisher test

        :return pd.DataFrame: contains:
          W: test statistic of WBF test
          W_low: 2.5% confidence boundary for W
          W_high: 97.5% confidence boundary for W
          p: p-value corresponding to W
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

    def fit_noparallel(self, n_iter=50, alpha=0.05):
        self._proc_init()
        results = []
        for i in np.arange(n_iter):
            results.append(self.map(self.n_samples_to_draw))
        return self.reduce(results, alpha=alpha)
