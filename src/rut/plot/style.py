import matplotlib.pyplot as plt
import matplotlib as mpl


def set_style(style='default'):
    """

    :param str style: [default, paper] set the style for output plots
    :return:
    """
    if style is 'default':
        plt.style.use('fivethirtyeight')
        mpl.rcParams['font.family'] = 'monospace'
    if style is 'paper':
        plt.style.use('default')  # mpl 2.0 defaults are good for papers
