import matplotlib.pyplot as plt


def inverted_curve(
        dependent, independent, ax=None, ylabel='dependent', xlabel='independent',
        **plot_kwargs):
    """

    :param np.array dependent: dependent varible (y-axis)
    :param np.array independent: independent variable (x axis), this is inverted in this
      plot. Useful, for example, for plotting dependency on cell number
    :param xlabel: x-axis label
    :param ylabel: y-axis label
    :param ax: axis

    :return matplotlib.axes._subplots.AxesSubplot: axis object containing plot
    """
    if ax is None:
        f, ax = plt.subplots(figsize=(3, 3))
    ax.plot(independent, dependent, **plot_kwargs)
    ax.invert_xaxis()
    _ = ax.set_xticks(dependent[::-2])  # makes ugly plots if there are too many ticks
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
