from itertools import chain
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


def p_value_vs_observation_size(gene_abundance, *p_values, ax=None, **kwargs):
    """

    :param pd.Series gene_abundance:
    :param [pd.Series] p_values: arbitrary number of pandas series to be plotted.
    :param mpl.Axes.axes ax: matplotlib axes object

    :return mpl.Axes.axes:
    """
    if ax is None:
        f, ax = plt.subplots(figsize=(3, 3))

    # sort p_values just in case, then concatenate
    concat_p_values = np.concatenate([p.loc[gene_abundance.index].values for p in p_values])
    concat_gene_abundance = np.concatenate([gene_abundance.values] * len(p_values))

    # create color vector
    colors = [mpl.colors.to_rgba(c) for c in plt.rcParams["axes.prop_cycle"].by_key()["color"]]
    cvector = np.vstack(chain([c] * len(p) for (c, p) in zip(colors, p_values)))

    # randomize order
    idx = np.random.choice(np.arange(concat_p_values.shape[0]), replace=False)

    # plot
    ax.scatter(concat_gene_abundance[idx], concat_p_values[idx], c=cvector[idx], **kwargs)

    return ax

