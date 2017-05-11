import unittest
import nose2
import numpy as np
import pandas as pd
from rut.differential_expression import mannwhitneyu, kruskalwallis
from rut.testing import empirical_variance
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
        print(mwu.fit(8, 8))


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


if __name__ == "__main__":
    nose2.main()
