import numpy as np
from statsmodels.sandbox.stats.multicomp import multipletests
from scipy.stats.mstats import count_tied_groups, rankdata
from rut.differential_expression import differential_expression
import rut.misc
import pandas as pd


class WelchsT(differential_expression.DifferentialExpression):

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
        Welch's T-test test between classes defind by global variables data
        and splits

        Designed to be mapped to a multiprocessing pool. Draws a sample of size n from
        each class

        :param int n: number of observations to draw with replacement per class sampled
          from data
        :return np.ndarray: n features x 2 array with columns of test statistics and
          p-values
        """

        def wtt(x, y, n_x, n_y):
            x_mu = x.mean(axis=0)
            y_mu = y.mean(axis=0)
            x_var = np.var(x, axis=0)
            y_var = np.var(y, axis=0)
            z = (x_mu - y_mu) / np.sqrt((x_var / n_x) + (y_var / n_y))
            return z

        complete_data = cls.get_shared_data()
        array_splits = cls.get_shared_splits()
        assert array_splits.shape[0] == 1  # only have two classes
        xy = cls._draw_sample_with_replacement(complete_data, n, array_splits)

        if xy.ndim == 1:
            xy = xy[:, np.newaxis]

        # calculate test statistic
        n_x = array_splits[0]
        n_y = xy.shape[0] - n_x
        return wtt(xy[:n_x, :], xy[n_x:, :], n_x, n_y)

    def _reduce(self, results, alpha=0.05):
        """
        reduction function for Welch's T-test test that processes the results
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
        results[np.isnan(results)] = 0  # todo fix this
        ci = rut.misc.confidence_interval(results)
        # p = rut.misc.z_to_p(results.sum(axis=0) / np.sqrt(results.shape[0]))[:, None]
        results = pd.DataFrame(
            data=np.concatenate([np.mean(results, axis=0)[:, None], ci], axis=1),
            index=self._features,
            columns=['t', 't_low', 't_high'])
        results['p'] = rut.misc.z_to_p(results['t'])  # mean z

        # add multiple-testing correction
        results['q'] = multipletests(results['p'], alpha=alpha, method='fdr_by')[1]

        results = results.sort_values('q')
        results.iloc[:, 1:4] = np.round(results.iloc[:, 1:4], 2)
        return results

    def fit(self, n_iter=50, n_processes=None, alpha=0.05):
        """
        Carry out a Welch's T-test

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

    def fit_noparallel(self, n_iter=50, alpha=0.05):
        self._proc_init()
        results = []
        for i in np.arange(n_iter):
            results.append(self._map(self.n_samples_to_draw))
        return self._reduce(results, alpha=alpha)
