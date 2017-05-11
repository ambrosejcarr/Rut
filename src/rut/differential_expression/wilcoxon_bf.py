import numpy as np
from statsmodels.sandbox.stats.multicomp import multipletests
from scipy.stats.mstats import count_tied_groups, rankdata
from rut.differential_expression import differential_expression


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

        complete_data = cls.get_shared_data()
        array_splits = cls.get_shared_splits()
        assert array_splits.shape[0] == 1  # only have two classes
        xy = cls._draw_sample_with_replacement(complete_data, n, array_splits)
        # calculate U for x
        if xy.ndim == 1:
            xy = xy[:, np.newaxis]
        ranks = rankdata(xy, axis=0)
        del xy  # memory savings
        nt = 2 * n
        U = ranks[:n].sum(0) - n * (n + 1) / 2.

        # get mean value
        mu = (n ** 2) / 2.

        # get smaller u (convention) for reporting only
        u = np.amin([U, n ** 2 - U], axis=0)

        sigsq = np.ones(ranks.shape[1]) * (nt ** 3 - nt) / 12.

        for i in np.arange(len(sigsq)):
            ties = count_tied_groups(ranks[:, i])
            sigsq[i] -= np.sum(v * (k ** 3 - k) for (k, v) in ties.items()) / 12.
        sigsq *= n ** 2 / float(nt * (nt - 1))

        # ignore division by zero warnings; they are properly dealt with by this test.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # note that in the case that sigsq is zero, and mu=U, this line will produce
            # -inf. However, this z-score should be zero, as this is an artifact of
            # continuity correction (subtraction of 1/2)
            z = (U - 1 / 2. - mu) / np.sqrt(sigsq)

        # correct infs
        z[sigsq == 0] = 0

        return np.vstack([u, z]).T


    def mwu_reduce(self, results, alpha=0.05):
        """
        reduction function for Mann-Whitney U-test that processes the results from
        mw_map into a results object

        :param list results: output from mw_map function, a list of np.array objects
          containing U-scores and z-scores.
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
        ci = misc.confidence_interval(results[:, :, 1])

        results = pd.DataFrame(
            data=np.concatenate([np.median(results, axis=0), ci], axis=1),
            index=self._features,
            columns=['U', 'z_approx', 'z_lo', 'z_hi'])

        # calculate p-values for median z-score
        results['p'] = misc.z_to_p(results['z_approx'])

        # add multiple-testing correction
        results['q'] = multipletests(results['p'], alpha=alpha, method='fdr_by')[1]

        results = results.sort_values('q')
        results.iloc[:, 1:4] = np.round(results.iloc[:, 1:4], 2)
        return results


    def fit(self, n_iter=50, n_processes=None, alpha=0.05):
        """
        Carry out a Mann-Whitney U-test

        :return:
        """
        self.result_ = self.run(
            n_iter=n_iter,
            n_processes=n_processes,
            fmap=self.mwu_map,
            freduce=self.mwu_reduce,
            freduce_kwargs=dict(alpha=alpha)
        )

        return self.result_
