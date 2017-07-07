import numpy as np
from statsmodels.sandbox.stats.multicomp import multipletests
from rut.differential_expression import differential_expression
import rut.misc
import pandas as pd
from rut.sample import Sampled


class WelchsT(Sampled, differential_expression.DifferentialExpression):
    """
    Welch's T-test test over down-sampled data from two groups of cells.
    """


    def __init__(self, *args, **kwargs):
        """

        :param pd.DataFrame | np.array data: m observations x p features array or
          dataframe
        :param np.ndarray labels: m x 1 condition labels that separate cells into units of
          comparison
        :param bool is_sorted: if True, no sorting is done of data or labels
        :param int max_obs_per_sample: hard ceiling on the number of observations to
          take for each sample. Useful for constraining memory usage

        """
        super().__init__(*args, **kwargs)
        if self._labels is None:
            raise ValueError('Labels are required for Welch\'s t-test')
        elif np.unique(self._labels).shape[0] != 2:
            raise ValueError(
                'Labels must contain only two categories for WelchsT testing. '
                'Please use KruskalWallis for poly-sample comparisons')

    @classmethod
    def map(cls, n):
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
        return wtt(xy[:n, :], xy[n:, :], n, n)  # samples are always size n

    def reduce(self, results, alpha=0.05):
        """
        reduction function for Welch's T-test test that processes the results
        from self._map into a results object

        :param list results: output from mw_map function, a list of np.array objects
          containing z-scores
        :param float alpha: acceptable type-I error rate for BY (negative) FDR correction

        :return pd.DataFrame: contains:
          t: median z-score across iterations
          t_low: 2.5% confidence boundary for t-score
          t_high: 97.5% confidence boundary for t-score
          p: p-value corresponding to t
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
        """Calculate a bootstrapped Welch's T-test

        :param int n_iter: number of sampling iterations to run
        :param int n_processes: number of processes to use in the pool (default = number
          available to runtime environment)
        :param float alpha: allowable type-I error

        :return pd.DataFrame: contains:
          t: median z-score across iterations
          t_low: 2.5% confidence boundary for t-score
          t_high: 97.5% confidence boundary for t-score
          p: p-value corresponding to t
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
