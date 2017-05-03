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

    def __init__(self, p_values_list, condition):
        """
        :param [pd.Series] p_values_list: list of pandas series containing p-values
        :param pd.Series[bool] condition: true condition labels
        """
        self.p_values_list = p_values_list
        self.condition = condition

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

