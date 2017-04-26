

def save_result(result_object, filename):
    """save a rut results object to file

    :param pd.DataFrame result_object: output of mannwhitneyu or kruskalwallis
    :param str filename: filename for saved object
    :return None: saves the data to file
    """
    result_object.to_csv(filename)


