import numpy as np
import rut.sample


class EmpiricalVariance(rut.sample.Sampled):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.result_ = None

    @classmethod
    def _map(cls, n):
        complete_data = cls.get_shared_data()
        array_splits = cls.get_shared_splits()
        assert array_splits.shape[0] == 1  # only have two classes
        xy = cls._draw_sample_with_replacement(complete_data, n, array_splits)

        x = xy[:n, :]
        y = xy[n:, :]

        mu = xy.mean(axis=0)  # all samples are equal-sized

        sigma_x = x.var(axis=0)
        sigma_y = y.var(axis=0)
        return sigma_x - sigma_y / mu

    @staticmethod
    def _reduce(results):
        return np.vstack(results)

    def fit(self, n_iter=50, n_processes=None):
        self.result_ = self.run(
            n_iter=n_iter,
            n_processes=n_processes,
            fmap=self._map,
            freduce=self._reduce,
        )

        return self.result_
