import numpy as np
import pandas as pd


class StatisticsConfusionMatrix:

    def __init__(self, p_values, condition):
        """

        :param pd.Series p_values: actual class labels
        :param pd.Series[bool] condition: true condition labels
        """
        self._p_values = p_values
        self._prediction = condition.loc[p_values.index]
        self.confusion_ = {}

    def _calculate_confusion(self, alpha):
        significant = np.less(self._p_values, alpha)
        self.confusion_[alpha] = dict(
            tp=np.logical_and(significant, self._prediction).sum(),
            fp=np.logical_and(significant, ~self._prediction).sum(),
            fn=np.logical_and(~significant, self._prediction).sum(),
            tn=np.logical_and(~significant, ~self._prediction).sum(),
        )
        return self.confusion_[alpha]

    def precision(self, alpha):
        try:
            c = self.confusion_[alpha]
        except KeyError:
            c = self._calculate_confusion(alpha)
        return c['tp'] / (c['tp'] + c['fp'])

    def recall(self, alpha):
        try:
            c = self.confusion_[alpha]
        except KeyError:
            c = self._calculate_confusion(alpha)
        return c['tp'] / (c['tp'] + c['fn'])

    def receiver_operator_characteristic(self, alpha_levels=20):
        alphas = np.linspace(0., 1., alpha_levels)
        precision = []
        fpr = []
        for a in alphas:
            self._calculate_confusion(a)
            precision.append(self.precision(a))
            fpr.append(1. - self.recall(a))
        return precision, fpr

    def f_score(self, alpha):
        p = self.precision(alpha)
        r = self.recall(alpha)
        return p + r / (p * r)


class AnalysisByCellNumber:

    def __init__(self, p_values_list, condition, cell_number_labels=None):
        """
        :param [pd.Series] p_values_list: list of pandas series containing p-values
        :param pd.Series[bool] condition: true condition labels
        """
        self.p_values_list = p_values_list
        self.condition = condition
        if cell_number_labels is not None:
            self.cell_number_labels = cell_number_labels
        else:
            self.cell_number_labels = np.arange(p_values_list[0].shape)

    def recall_vs_cell_number(self, alpha):
        recall_values = []
        for p_values in self.p_values_list:
            recall_values.append(
                StatisticsConfusionMatrix(p_values, self.condition).recall(alpha))
        return recall_values

    def precision_vs_cell_number(self, alpha):
        precision_values = []
        for p_values in self.p_values_list:
            precision_values.append(
                StatisticsConfusionMatrix(p_values, self.condition).precision(alpha))
        return precision_values

    def plot_precision_by_cell_number(self, alpha=0.05, **kwargs):
        import rut.plot.differential_expression.diagnostic_curves as dc
        precision = self.precision_vs_cell_number(alpha)
        return dc.inverted_curve(
            precision, self.cell_number_labels, ylabel='precision', xlabel='cell number',
            **kwargs
        )

    def plot_recall_by_cell_number(self, alpha=0.05, **kwargs):
        import rut.plot.differential_expression.diagnostic_curves as dc
        recall = self.recall_vs_cell_number(alpha)
        return dc.inverted_curve(
            recall, self.cell_number_labels, ylabel='recall', xlabel='cell number',
            **kwargs
        )


class MultiTestComparison:

    def __init__(self, test_results, condition):
        """

        :param dict test_results:
        :param true result labels condition:
        """
        pass

    def tp_venn_diagram(self):
        # number of comparisons should be at most three; pick best performing methods
        # to do these comparisons.
        raise NotImplementedError

    def fp_venn_diagram(self):
        raise NotImplementedError


class DataDiagnostics:

    def __init__(self, data):
        """
        This class contains several tests for data distribution which reveal data problems
        that suggest the necessity of non-parametric testing.
        """
        self._data = data

    def residual_variance_by_factor(self):
        """
        Display residual differences in variation across samples which would not be
        captured by simple transformations of the mean (need to transform variance too!)
        :return:
        """
        raise NotImplementedError

    def display_factor_effect(self):
        """
        scatterplot of mean vs variance across factors?
        :return:
        """
        raise NotImplementedError

    def library_size_distribution(self, ax=None):
        import rut.plot.diagnostics
        library_size = self._data.sum(axis=1)
        rut.plot.diagnostics.smoothed_histogram(
            library_size, ax=ax, xlabel='molecules per cell')
        return ax

    # todo what else should I be adding here? q-q plots? other?
