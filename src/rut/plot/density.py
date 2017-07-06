import matplotlib.pyplot as plt
import statsmodels.nonparametric.api as smnp
import rut.plot.annotate_axis


def smoothed_histogram(data, xlabel=None, ax=None, figsize=(3, 3), kde_kwargs=None):
    if kde_kwargs is None:
        kde_kwargs = {}
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    bins = max(25, len(data) // 100)
    height, *_ = ax.hist(data, bins=bins, alpha=0.5)
    kde = smnp.KDEUnivariate(data.astype(float))
    kde.fit(**kde_kwargs)
    scale_factor = height.max() / kde.density.max()
    x, y = kde.support, kde.density
    ax.plot(x, y * scale_factor, linewidth=2)

    rut.plot.annotate_axis.annotate_axis(ax=ax, xlabel=xlabel, ylabel='Cells',
                                         title='Library Size Distribution')

    return ax
