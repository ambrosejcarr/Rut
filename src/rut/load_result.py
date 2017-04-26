import pandas as pd


def load_result(filename):
    """load a rut results object from file

    :param str filename: name of the saved results objects
    :return pd.DataFrame: loaded results object
    """
    return pd.read_csv(filename, index_col=0, header=0)
