import numpy as np
import matplotlib.pyplot as plt


def volcano(a, b, results, sampling_percentile=10, ax=None):
    """volcano is -log10 p-value against mean expression

    :param a:
    :param b:
    :param results:
    :param ax:
    """

    if ax is None:
        ax = plt.gca()

    sig = results['q'] < 0.1
    sig_up = sig & (results['z_approx'] > 0)
    sig_dn = sig & (results['z_approx'] < 0)

    # find the sampling value
    sampling_value = min(
        np.percentile(g, sampling_percentile)
        for g in [a.sum(axis=1), b.sum(axis=1)])

    # correct library contributions for sampling
    an = a.mul(sampling_value / a.sum(axis=1), axis=0)
    bn = b.mul(sampling_value / b.sum(axis=1), axis=0)

    a_mu = np.mean(an, axis=0)
    b_mu = np.mean(bn, axis=0)

    mean_expression = (a_mu + b_mu) / 2

    fc = b_mu - a_mu
    valid = ~(np.isnan(fc) | np.isinf(fc))
    magnitude = -10 * np.log10(results['p'])
    ax.plot(mean_expression[valid & sig_up], fc[valid & sig_up], linewidth=0, marker='o',
            markersize=2, zorder=2, label='a < b; q < 0.1')
    ax.plot(mean_expression[valid & sig_dn], fc[valid & sig_dn], linewidth=0, marker='o',
            markersize=2, zorder=1, label='a > b; q < 0.1')
    ax.plot(mean_expression[valid & (~sig)], fc[valid & (~sig)], linewidth=0, marker='o',
            markersize=2, c='k', zorder=0, label='Not DE')

    # plot up to 99.9th percentile
    xmax = np.percentile(mean_expression[valid], 99.9)
    ymax = np.percentile(np.abs(fc[valid]), 99.9)
    ax.set_ylim((-ymax, ymax))
    ax.set_xlim((0, xmax))

    # label axes
    ax.set_xlabel('$\mu$ molecules per cell', fontname='DejaVu Sans')
    ax.set_ylabel('$\mu_{a} - \mu_{b}$')
    ax.set_title('RUT Volcano Plot')

    ax.legend(bbox_to_anchor=[0.98, 0.5], loc='center left', markerscale=2)

    return ax
