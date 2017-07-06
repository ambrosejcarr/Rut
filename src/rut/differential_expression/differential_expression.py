from rut import sample


class DifferentialExpression(sample.Sampled):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.result_ = None

    def volcano_plot(self):
        """

        :return:
        """
        pass

    # todo think about pickling or similar; don't necessarily want to save data...
    def save(self, filename):
        """

        :return:
        """
        if self.result_ is None:
            raise ValueError('run fit() before results can be saved.')
        if not filename.endswith('.gz'):
            filename += '.gz'
        self.result_.to_csv(filename, compression='gzip')

    def load(self, filename):
        pass
