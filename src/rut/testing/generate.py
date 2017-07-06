import os
import numpy as np
import pickle
import pandas as pd
from collections import defaultdict
from collections.abc import Callable
from subprocess import Popen, PIPE
from rut.sample import Sampled


class SyntheticTest:

    def __init__(self, data, effects):
        """

        :param str data: filename pointing to a .csv file containing synthetic data over
          two classes, used to test various methods
        :param str effects:  filename pointing to a .p file containing the effects that
          have been induced in data

        """
        self.data_file = data
        self.effects = effects

    @classmethod
    def from_dataset(cls, data, labels, save, additional_downsampling=1):
        """

        :param pd.DataFrame data: dataframe containing m cells x n genes
        :param str save: file stem; location to save data
        :param list labels: list of two labels to give the original and synthetic
           populations
        :param additional_downsampling: downsampling to add to the sample to induce a
          change in the sampling rate, as is often observed in single-cell data

        :return SyntheticTest:  instance of a synthetic test class
        """

        _ = cls._synthesize(data, labels, save, additional_downsampling)
        data_name = save + '_ds_{:.2f}.csv'.format(additional_downsampling)
        effects_name = save + '_ds_{:.2f}_labels.p'.format(additional_downsampling)
        return cls(data_name, effects_name)

    @staticmethod
    def _synthesize(data, labels, save=None, additional_downsampling=1.):
        """
        return a synthetically adjusted sample to pair with the input sample for
        DE analysis

        the synthetic sample will have:
        50% non-de genes
        25% genes up, 25% genes down
        - 5% each 10% up/down
        - 5% each 20% up/down
        - 5% each 30% up/down
        - 5% each 40% up/down
        - 5% each 50% up/down

        genes from the input_sample will be sorted by expression and sorted evenly
        into each of these categories to ensure that library size is maximally spanned
        by each category

        :param pd.DataFrame data: dataframe containing m cells x n genes
        :param str save: file stem; location to save data
        :param list labels: list of two labels to give the original and synthetic
           populations
        :param additional_downsampling: downsampling to add to the sample to induce a
          change in the sampling rate, as is often observed in single-cell data


        :return pd.DataFrame: m x n adjusted count matrix
        """

        if not isinstance(data, pd.DataFrame):
            raise TypeError('data must be passed as a pandas DataFrame object')

        gene_sums = data.sum(axis=0).sort_values()
        gene_iterator = iter(gene_sums.index)
        effects = defaultdict(list)
        upsample_magnitudes = [.1, .2, .3, .4, .5]
        downsample_magnitudes = [-.1, -.2, -.3, -.4, -.5]
        effect_types = [0.] * 10 + upsample_magnitudes + downsample_magnitudes

        # partition the genes
        while True:
            try:
                for e in effect_types:
                    gene = next(gene_iterator)
                    effects[e].append(gene)
            except StopIteration:
                break

        # store synthesized data
        synthetic = []

        # iterate over upsampling magnitudes
        for upsample_magnitude in upsample_magnitudes:
            # count the number of molecules that need to be added or subtracted
            indices = effects[upsample_magnitude]
            add_counts = np.round(gene_sums[indices] * upsample_magnitude)

            # add that number of molecules with uniform probability across genes
            for index, n in add_counts.items():  # (gene, count)
                localized_counts = np.random.multinomial(n, np.ones(data.shape[0]) /
                                                         data.shape[0])
                synthetic.append(
                    pd.DataFrame(data[index] + localized_counts, index=data.index,
                                 columns=[index]))

        # iterate over downsampling magnitudes
        for downsample_magnitude in downsample_magnitudes:
            indices = effects[downsample_magnitude]
            adjusted = data[indices].mul(1 - downsample_magnitude)
            p = np.random.sample(adjusted.shape)
            synthetic.append(
                pd.DataFrame(np.floor(adjusted) + (adjusted % 1 > p).astype(int),
                             index=data.index, columns=indices))

        synthetic = pd.DataFrame(pd.concat(synthetic + [data[effects[0]]], axis=1))
        synthetic = synthetic[data.columns]  # reorder columns

        # add any additional downsampling
        if additional_downsampling < 1.:
            synthetic = synthetic.mul(additional_downsampling)
            p = np.random.sample(synthetic.shape)
            synthetic = (synthetic % 1 > p).astype(int) + np.floor(synthetic)

        # merge the results together
        labeled_index =  [labels[0]] * data.shape[0] + [labels[1]] * synthetic.shape[0]
        merged = pd.DataFrame(
            pd.concat([data, synthetic], axis=0))
        merged.index = labeled_index

        # save the results
        if save is not None:
            data_name = save + '_ds_{:.2f}.csv'.format(additional_downsampling)
            standards_name = save + '_ds_{:.2f}_labels.p'.format(
                additional_downsampling)
            merged.to_csv(data_name)
            with open(standards_name, 'wb') as f:
                pickle.dump(effects, f)

        return merged, effects

    def test_method(self, function_or_script, outfile_name=None):
        """

        :param str | Callable function_or_script:  if a str script is passed, it should
          be an R script whose function is to save a file which is loadable by python
          and contains a column of p-values with an index of genes. If a callable is
          passed, it should return a similar structure directly.
        :param outfile_name:

        :return pd.DataFrame: DataFrame containing test results, with at least one column
          containing p-values
        """

        # execute script
        if isinstance(function_or_script, str):
            if outfile_name is None:
                raise ValueError('must pass outfile name when passing script')
            args = ['RScript', os.path.expanduser(function_or_script), self.data_file,
                    outfile_name]
            p = Popen(args, stderr=PIPE, stdout=PIPE)
            out, err = p.communicate()
            if err:
                print(err.decode())
            if out:
                print(out.decode())

            # load the file
            results = pd.read_csv(outfile_name, index_col=0)

        else:  # run module
            data = pd.read_csv(self.data_file, index_col=0)
            labels = data.index
            prototype = function_or_script(data, labels)
            results = prototype.fit()
            if outfile_name is not None:
                results.to_csv(outfile_name)

        return results

    def test_all_de_methods(self):
        raise NotImplementedError  # todo implement me


class SimpleTest(SyntheticTest):

    def __init__(self, *args, **kwargs):
        """

        :param str data: filename pointing to a .csv file containing synthetic data over
          two classes, used to test various methods
        :param str effects:  filename pointing to a .p file containing the effects that
          have been induced in data

        """
        super().__init__(*args, **kwargs)

    @staticmethod
    def downsample(data, downsampling_value=0.5, keep_fraction_genes=1.):
        """

        :param pd.DataFrame data: data to downsample
        :param float downsampling_value: percentage of data to retain
        :param float keep_fraction_genes: percentage of genes to retain. Note that in normal cirumstances,
          one would want to avoid doing this type of subsetting as it changes the data distribution. However,
          in this synthetic case, it is acceptable to do this to speed up the data generation.
        :return:
        """
        # normalize the data, then draw a sample
        if keep_fraction_genes < 1.:
            idx = np.floor(np.linspace(0, data.shape[1] - 1, np.floor(data.shape[1] * 0.5).astype(int))).astype(int)
            data = data.iloc[:, idx]
        comparison = data.mul(downsampling_value)
        comparison = pd.DataFrame(
            Sampled._draw_sample(comparison.values),
            index=data.index,
            columns=data.columns)
        labeled_index = [0] * data.shape[0] + [1] * comparison.shape[0]
        merged = pd.concat([data, comparison], axis=0)
        merged.index = labeled_index

        # effects are all 0
        effects = {0: list(merged.columns)}

        return merged, effects

    @classmethod
    def from_dataset(cls, data, save, additional_downsampling=0.5, keep_fraction_genes=1., labels=None):
        """

        :param pd.DataFrame data: dataframe containing m cells x n genes to use as basis for synthetic data generation
        :param str save: file stem; location to save data
        :param additional_downsampling: downsampling to add to the sample to induce a
          change in the sampling rate, as is often observed in single-cell data
        :param keep_fraction_genes: retain this fraction of genes when generating synthetic data
        :param list labels: Not used

        :return SyntheticTest:  instance of a synthetic test class
        """

        merged, effects = cls.downsample(data, additional_downsampling, keep_fraction_genes)
        data_name = save + '_ds_{:.2f}.csv'.format(additional_downsampling)
        effects_name = save + '_ds_{:.2f}_labels.p'.format(additional_downsampling)
        merged.to_csv(data_name)
        with open(effects_name, 'wb') as f:
            pickle.dump(effects, f)

        return cls(data_name, effects_name)