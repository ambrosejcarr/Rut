import os
import unittest
import nose2
import numpy as np
import pandas as pd
from rut.differential_expression import mannwhitneyu, kruskalwallis, wilcoxon_bf, welchs_t
from rut.testing import external_comparisons
from rut.testing import empirical_variance, generate
from rut import score_feature_magnitude, cluster



class TestKruskalWallis(unittest.TestCase):

    def test_synthetic(self):
        x = np.array([
            np.random.randint(0, 5, 10),  # lower than y
            np.ones(10) * 100,  # large, takes up most of library size normalization
            np.random.randint(5, 10, 10)  # higher than y
        ]).T

        y = np.array([
            np.random.randint(5, 10, 10),  # higher than y
            np.ones(10) * 100,  # takes up most of library size normalization
            np.random.randint(0, 5, 10)  # lower than y
        ]).T

        data = pd.DataFrame(
            data=np.concatenate([x, y], axis=0)
        )
        labels = np.concatenate([np.ones(10), np.zeros(10)], axis=0)
        kw = kruskalwallis.KruskalWallis(data, labels)
        print(kw.fit())


class TestMannWhitneyU(unittest.TestCase):

    def test_synthetic(self):
        x = np.array([
            np.random.randint(0, 5, 10),  # lower than y
            np.ones(10) * 100,  # large, takes up most of library size normalization
            np.random.randint(5, 10, 10)  # higher than y
        ]).T

        y = np.array([
            np.random.randint(5, 10, 10),  # higher than y
            np.ones(10) * 100,  # takes up most of library size normalization
            np.random.randint(0, 5, 10)  # lower than y
        ]).T

        data = pd.DataFrame(
            data=np.concatenate([x, y], axis=0)
        )
        labels = np.concatenate([np.ones(10), np.zeros(10)], axis=0)
        mwu = mannwhitneyu.MannWhitneyU(data, labels)
        print(mwu.fit())

    def test_at_scale(self):
        data = pd.read_table(
            '/Users/ambrose/google_drive/manuscripts/rut/data/r_comparisons'
            '/cluster_21_vs_28_for_R.txt', index_col=0).T

        half = int(data.shape[0] / 2)
        labels = np.array((['21'] * half) + (['21_adj'] * half))
        mwu = mannwhitneyu.MannWhitneyU(data, labels)
        print(mwu.fit())


class TestWelchsT(unittest.TestCase):

    def test_synthetic(self):
        n = 100
        x = np.array([
            np.random.randint(0, 5, n),  # lower than y
            np.ones(n) * 100,  # large, takes up most of library size normalization
            np.random.randint(5, 10, n)  # higher than y
        ]).T

        y = np.array([
            np.random.randint(5, 10, n),  # higher than y
            np.ones(n) * 100,  # takes up most of library size normalization
            np.random.randint(0, 5, n)  # lower than y
        ]).T

        data = pd.DataFrame(
            data=np.concatenate([x, y], axis=0)
        )
        labels = np.concatenate([np.ones(n), np.zeros(n)], axis=0)
        wtt = welchs_t.WelchsT(data, labels)
        print(wtt.fit_noparallel())

    def test_at_scale(self):
        data = pd.read_table(os.path.expanduser(
            '~/google_drive/manuscripts/rut/data/r_comparisons'
            '/cluster_21_vs_synthetic_ds_0.75.txt'), index_col=0).T

        half = int(data.shape[0] / 2)
        labels = np.array((['21'] * half) + (['21_adj'] * half))
        wtt = welchs_t.WelchsT(data, labels)
        print(wtt.fit())

    def test_at_scale_noparallel(self):
        data = pd.read_table(os.path.expanduser(
            '~/google_drive/manuscripts/rut/data/r_comparisons'
            '/cluster_21_vs_synthetic_ds_0.75.txt'), index_col=0).T

        half = int(data.shape[0] / 2)
        labels = np.array((['21'] * half) + (['21_adj'] * half))
        wtt = welchs_t.WelchsT(data, labels)
        print(wtt.fit())


class TestWilcoxonBF(unittest.TestCase):

    def test_synthetic(self):
        n = 100
        x = np.array([
            np.random.randint(0, 5, n),  # lower than y
            np.ones(n) * 100,  # large, takes up most of library size normalization
            np.random.randint(5, 10, n)  # higher than y
        ]).T

        y = np.array([
            np.random.randint(5, 10, n),  # higher than y
            np.ones(n) * 100,  # takes up most of library size normalization
            np.random.randint(0, 5, n)  # lower than y
        ]).T

        data = pd.DataFrame(
            data=np.concatenate([x, y], axis=0)
        )
        labels = np.concatenate([np.ones(n), np.zeros(n)], axis=0)
        wbf = wilcoxon_bf.WilcoxonBF(data, labels)
        print(wbf.fit())

    def test_at_scale(self):
        data = pd.read_table(os.path.expanduser(
            '~/google_drive/manuscripts/rut/data/r_comparisons'
            '/cluster_21_vs_synthetic_ds_0.75.txt'), index_col=0).T

        half = int(data.shape[0] / 2)
        labels = np.array((['21'] * half) + (['21_adj'] * half))
        wbf = wilcoxon_bf.WilcoxonBF(data, labels)
        print(wbf.fit())

    def test_at_scale_noparallel(self):
        data = pd.read_table(os.path.expanduser(
            '~/google_drive/manuscripts/rut/data/r_comparisons'
            '/cluster_21_vs_synthetic_ds_0.75.txt'), index_col=0).T

        half = int(data.shape[0] / 2)
        labels = np.array((['21'] * half) + (['21_adj'] * half))
        wbf = wilcoxon_bf.WilcoxonBF(data, labels)
        print(wbf.fit_noparallel())

    def test_author_example(self):
        from scipy.stats.mstats import rankdata
        x = np.array([1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 4, 1, 1])[:, None]
        y = np.array([3, 3, 4, 3, 1, 2, 3, 1, 1, 5, 4])[:, None]

        def empirical_variance(within_rank, total_rank, mean_total_rank, n, N):
            """

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

        res = wbf(np.concatenate([x, y], axis=0), len(x), len(y))
        print(res)
        self.assertAlmostEqual(np.round(res[0], 3), 3.137)


class TestScoreFeatureMagnitude(unittest.TestCase):

    def test_synthetic(self):
        labels = np.array(([0] * 5) + ([1] * 5))
        data = pd.DataFrame(
            np.random.randint(0, 10, 100).reshape(10, 10),
            index=list('abcdefghij'),
            columns=list('klmnopqrst')
        )

        feature_groups = {'ten': list('klmnz'), 'gtf': list('znm'), 'ded': list('pqe'),
                          'you_should_not_see_me': list('xyz')}
        sfm = score_feature_magnitude.ScoreFeatureMagnitudes(
            data, labels, feature_groups=feature_groups)
        print(sfm.fit(4, 4)['mean'])


class TestCluster(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        m1 = np.zeros(100)
        cov1 = np.diag(np.random.rand(100))
        m2 = np.zeros(100) + 0.5
        cov2 = np.diag(np.random.rand(100))
        m3 = np.ones(100)
        cov3 = np.diag(np.random.rand(100)) * 1.5

        cls.data_ = np.concatenate([
            np.random.multivariate_normal(m1, cov1, 100),
            np.random.multivariate_normal(m2, cov2, 100),
            np.random.multivariate_normal(m3, cov3, 100),
        ], axis=0)

    def test_cluster(self):
        # draw data from three different multivariate gaussians
        import matplotlib.pyplot as plt
        import rut.plot.matrix

        c = cluster.Cluster(self.data_)
        c.fit(4, 4)

        c.plot_confusion_matrix()
        plt.savefig('test_confusion_matrix.png', dpi=150)

        astr = c.score_association_strength()
        rut.plot.matrix.clustered_symmetric_matrix(astr, trim=0)
        plt.savefig('test_association_strength.png', dpi=150)

    def test_cluster_no_parallel(self):
        import matplotlib.pyplot as plt
        import rut.plot.matrix

        c = cluster.Cluster(self.data_)
        c.fit_no_parallelism(4, 4)

        c.plot_confusion_matrix()
        plt.savefig('test_confusion_matrix.png', dpi=150)

        astr = c.score_association_strength(n_iter=4)
        rut.plot.matrix.clustered_symmetric_matrix(astr, trim=0)
        plt.savefig('test_association_strength.png', dpi=150)


class TestEmpiricalVariance(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.x = np.array([
            np.random.randint(0, 5, 10),  # lower than y
            np.ones(10) * 100,  # large, takes up most of library size normalization
            np.random.randint(5, 10, 10)  # higher than y
        ]).T

        cls.y = np.array([
            np.random.randint(5, 10, 10),  # higher than y
            np.ones(10) * 100,  # takes up most of library size normalization
            np.random.randint(0, 5, 10)  # lower than y
        ]).T

        cls.data = pd.DataFrame(
            data=np.concatenate([cls.x, cls.y], axis=0)
        )
        cls.labels = np.concatenate([np.ones(10), np.zeros(10)], axis=0)

    def test_empirical_variance(self):
        ev = empirical_variance.EmpiricalVariance(self.data, self.labels)
        print(ev.fit())


class TestFisher(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.background = set(list('abcdefghijklmnopqrstuvwxyz'))
        cls.sets = [
            set(list('abcdefghijkl')),
            set(list('mnopqrstuvwx')),
            set(list('yz')),
            set(list('12345'))
        ]
        cls.test_sets = [
            set(list('abcd')),
            set(list('mnkj')),
            set(list('12345')),
            set(list('6789'))
        ]

    def test_fisher_test(self):
        import rut.fisher_test
        ft = rut.fisher_test.FisherTest(self.sets, self.background)
        print(list(ft.fit(s) for s in self.test_sets))


class TestGenerate(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        n = 500
        cls.x = np.array([
            np.random.randint(0, 5, n),  # lower than y
            np.ones(n) * 100,  # large, takes up most of library size normalization
            np.random.randint(5, 10, n)  # higher than y
        ]).T

        cls.y = np.array([
            np.random.randint(5, 10, n),  # higher than y
            np.ones(n) * 100,  # takes up most of library size normalization
            np.random.randint(0, 5, n)  # lower than y
        ]).T

        cls.data = pd.DataFrame(
            data=np.concatenate([cls.x, cls.y], axis=0)
        )
        # cls.labels = np.concatenate([np.ones(n), np.zeros(n)], axis=0)
        cls.tmpdir = os.environ['TMPDIR']

        cls.synth = generate.SyntheticTest.from_dataset(
            pd.DataFrame(cls.data), [0, 1], save=cls.tmpdir + 'test_synthetic',
            additional_downsampling=0.45)

    @unittest.skip('trivial development test')
    def test_synthetic(self):
        df, effects = generate.SyntheticTest._synthesize(
            pd.DataFrame(self.x), [0, 1], save=self.tmpdir + 'test_synthetic',
            additional_downsampling=0.5)
        # 0 should be ~2x 1
        print(df.loc[0, :].sum())
        print(df.loc[1, ].sum())

    def test_synthetic_mast(self):
        script = '~/projects/RutR/R/runMAST.R'
        results_filename = self.tmpdir + 'test_synthetic_mast_results.csv'
        res = self.synth.test_method(script, results_filename)
        print(res)

    def test_synthetic_edger(self):
        script = '~/projects/RutR/R/runEdgeR.R'
        results_filename = self.tmpdir + 'test_synthetic_edger_results.csv'
        bigger_data = np.hstack([self.x] * 4)  # edgeR complains if it doesn't have lots of data
        big_synth = generate.SyntheticTest.from_dataset(
            pd.DataFrame(bigger_data), [0, 1], save=self.tmpdir + 'test_synthetic',
            additional_downsampling=0.45)
        res = big_synth.test_method(script, results_filename)
        print(res)

    @unittest.skip('takes 60s to run on test data, fails due to small gene number')
    def test_synthetic_scde(self):
        scde_script = '~/projects/RutR/R/runSCDE.R'
        results_filename = self.tmpdir + 'test_synthetic_SCDE_results.csv'  # this will fail unless has 2000 genes
        res = self.synth.test_method(scde_script, results_filename)
        print(res)

    def test_synthetic_mwu(self):
        results_filename = self.tmpdir + 'test_synthetic_rmwu_results.csv'
        res = self.synth.test_method(
            mannwhitneyu.MannWhitneyU, results_filename)
        print(res)

    def test_synthetic_wbf(self):
        results_filename = self.tmpdir + 'test_synthetic_rwbf_results.csv'
        res = self.synth.test_method(
            wilcoxon_bf.WilcoxonBF, results_filename)
        print(res)

    def test_synthetic_wtt(self):
        results_filename = self.tmpdir + 'test_synthetic_rwtt_results.csv'
        res = self.synth.test_method(
            welchs_t.WelchsT, results_filename)
        print(res)

    def test_synthetic_kartik_binomial(self):
        results_filename = self.tmpdir + 'test_synthetic_binomial_results.csv'
        res = self.synth.test_method(
            external_comparisons.BinomialTest, results_filename)
        print(res)


class TestGenerateSimple(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        n = 500
        cls.x = np.array([
            np.random.randint(0, 5, n),  # lower than y
            np.ones(n) * 100,  # large, takes up most of library size normalization
            np.random.randint(5, 10, n)  # higher than y
        ]).T

        cls.tmpdir = os.environ['TMPDIR']

        cls.synth = generate.SimpleTest.from_dataset(
            pd.DataFrame(cls.x), save=cls.tmpdir + 'test_synthetic_simple',
            additional_downsampling=0.5, keep_fraction_genes=.5)

    def test_synthetic_simple_mwu(self):
        results_filename = self.tmpdir + 'test_synthetic_simple_rmwu.csv'
        res = self.synth.test_method(
            mannwhitneyu.MannWhitneyU, results_filename)
        print(res)


if __name__ == "__main__":
    nose2.main()
