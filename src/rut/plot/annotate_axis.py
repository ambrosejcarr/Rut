

def annotate_axis(
        ax, xlabel=None, ylabel=None, title=None, xtick_rotation=0, ytick_rotation=0,
        xticks=None, yticks=None, xticklabels=None, yticklabels=None):
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels)
    if xtick_rotation:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=xtick_rotation)
    if ytick_rotation:
        ax.set_yticklabels(ax.get_yticklabels(), rotation=ytick_rotation)
