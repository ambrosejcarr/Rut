from rut.resampled_nonparametric import *
import unittest
import nose2
import numpy as np


class TestMannWhitneyU(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # create some test data
        cls.x1 = (np.repeat(10, 1000) + np.random.rand(1000))[:, None]
        cls.x2 = (np.repeat(10, 1000) + np.random.rand(1000))[:, None]
        cls.y = (np.repeat(100, 1000) + np.random.rand(1000))[:, None]

    def test_sampling_value(self):
        res = find_sampling_value([self.x1, self.y], 10)
        print(res)

    def test_normalization(self):
        pass

    def test_obs_number(self):
        pass

    def test_sampling_function(self):
        pass

    def test_confidence_interval(self):
        pass

    def test_mannwhitneyu(self):
        mannwhitneyu(self.x1, self.x2)


if __name__ == '__main__':
    nose2.main()