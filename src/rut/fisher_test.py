from functools import reduce
from itertools import repeat
from scipy.stats import fisher_exact
import numpy as np
from contextlib import closing
from multiprocessing import Pool

class FisherTest:

    def __init__(self, prior_sets, background):
        """
        :param list(set) prior_sets: lists of genes annotated by functions
          (ideally, gene ontology)
        :param set background: all genes in your experiment; will be used to restrict
          against genes that were not observed in your experiment, which can skew the
          statistic.
        """
        # restrict to genes in set

        # need to build tables
        #       in gs | not
        # sig |   A   |  s - A  | s
        #  ns | g - A | N-g-s-A | N - s
        #         g      N - g    N
        #
        # where A is g & s

        self._prior_sets = [s.union(background) for s in prior_sets]
        self._background = background
        self._N = len(self._background)
        self._g = [len(s) for s in self._prior_sets]

    def fit(self, gene_set, n_processes=None):
        """

        :return:
        """
        # g = comparison gene set

        # for each gene set, get size of set in target, size of target
        s = len(gene_set)
        a = gene_set.intersection(self._g)
        A = s - a
        b = len(self._g) - a
        B = self._N - len(self._g) - b

        # then, test with a, A, b, B
        iterable = [np.array([[m_, M], [n_, self._N]]).T for m_, n_ in zip(m, self._n)]

        with closing(Pool(processes=n_processes)) as pool:
            results = pool.map(fisher_exact, iterable)

        return results
