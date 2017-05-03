import unittest
import nose2
import numpy as np
import pandas as pd
from rut.differential_expression import mannwhitneyu, kruskalwallis
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

    def test_cluster(self):
        data = pd.DataFrame(np.random.randint(0, 10, 10000).reshape(100, 100))
        c = cluster.Cluster(data)
        results = c.fit(4, 4)
        print(results)


if __name__ == "__main__":
    nose2.main()
