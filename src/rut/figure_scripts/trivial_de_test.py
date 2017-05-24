from sys import argv
import os
import pandas as pd
import rut.testing.generate
import rut.differential_expression.mannwhitneyu
import rut.testing.external_comparisons
import h5


def main(data, save, downsampling=0.5, keep_genes=0.5):
    """

    :param str | pd.DataFrame data:
    :param str save: name to save intermediate data
    :param float downsampling:
    :param float keep_genes:
    :return:
    """

    if isinstance(data, str):
        data = pd.read_csv(data, index_col=0)

    test = rut.testing.generate.SimpleTest.from_dataset(data, save, downsampling, keep_genes)

    methods = {
        'r-MWU': rut.differential_expression.mannwhitneyu.MannWhitneyU,
        'MWU': rut.testing.external_comparisons.MannWhitneyU,
        'MAST':  '~/projects/RutR/R/runMAST.R',
        # 'SCDE': '~/projects/RutR/R/runSCDE.R',  # not working right now. Go R. Wew.
        'edgeR': '~/projects/RutR/R/runEdgeR.R',
        'binomial': rut.testing.external_comparisons.BinomialTest,
    }

    # dump data into temporary directory
    temp_name = os.environ['TMPDIR'] + '_%s_trivial_de_test.csv'

    # test the methods
    results = {}
    for method, module in methods.items():
        results[method] = test.test_method(module, temp_name % method)

    h5a = h5.Archive(save + '.h5')
    for method, result in results.items():
        h5a.save(result, method)


if __name__ == "__main__":
    if any(h in argv for h in ['-h', '--help']):
        print('usage: python3 trivial_de_test.py <data> <h5stem> <downsampling_value> <fraction_of_genes_to_use>')
    elif len(argv) != 5:
        print('usage: python3 trivial_de_test.py <data> <h5stem> <downsampling_value> <fraction_of_genes_to_use>')
    main(argv[1], argv[2], float(argv[3]), float(argv[4]))