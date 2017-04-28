from rut.mannwhitneyu import mannwhitneyu
from rut import sampling
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


class TestSampler(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.x = (np.repeat(10, 1000) + np.random.rand(1000)).reshape(10, 100)
        cls.y = (np.repeat(100, 1000) + np.random.rand(1000)).reshape(10, 100)
        cls.labels = np.concatenate([np.ones(10), np.zeros(10)])
        cls.data = np.vstack([cls.x, cls.y])
        print(cls.data.shape)

    def test_sampler(self):

        def fmap_test(n):
            global shared_data
            data = np.ctypeslib.as_array(shared_data)
            sample = sampling.draw_sample(data, n)
            return sample.sum()

        def freduce_test(results):
            return sum(results)

        sp = sampling.Sampler(self.data, fmap_test, freduce_test, labels=self.labels)
        print(sp.run(4))



if __name__ == '__main__':
    nose2.main()