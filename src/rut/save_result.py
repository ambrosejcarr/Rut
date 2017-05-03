

def save_result(result_object, filename):
    """save a rut results object to file

    :param pd.DataFrame result_object: output of mannwhitneyu or kruskalwallis
    :param str filename: filename for saved object
    :return None: saves the data to file
    """
    if not filename.endswith('.gz'):
        filename += '.gz'
    result_object.to_csv(filename, compression='gzip')


