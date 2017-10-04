import logging
import numpy as np
import utils


logger = logging.getLogger(__name__)


class DataSet(object):

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def batches_generator(self, batch_size=100):
        for b_x, b_y in utils.generate_batches(self.X, self.Y, batch_size):
            yield DataSet(b_x, b_y)


def read_dataset(x_filename, y_filename, transposed=True):
    X = np.loadtxt(x_filename, delimiter=',', unpack=transposed)
    Y = np.loadtxt(y_filename, delimiter=',', unpack=transposed)
    return DataSet(X, Y)


class FoldedDataSet(object):
    def __init__(self, folds_dir, n_folds, transposed=False):
        logger.debug('Reading dataset with %d folds', n_folds)
        self.folds = []
        for fold in xrange(n_folds):
            logger.debug('Reading fold %d...', fold)
            self.folds.append(read_dataset('%s/y_%d.csv' % (folds_dir, fold),
                                           '%s/b_%d.csv' % (folds_dir, fold),
                                           transposed))


    def batches_generator(self, batch_size=100):
        fold_generators = map(lambda fold: fold.batches_generator(batch_size), self.folds)

        while len(fold_generators) > 0:
            random_fold = np.random.randint(len(fold_generators))
            try:
                yield fold_generators[random_fold].next()
            except StopIteration:
                del fold_generators[random_fold]


class ShuffleFoldedDataSet(DataSet):
    def __init__(self, folds_dir, n_folds, transposed=False):
        logger.debug('Reading dataset with %d folds', n_folds)
        folds = []
        for fold in xrange(n_folds):
            logger.debug('Reading fold %d...', fold)
            folds.append(read_dataset('%s/y_%d.csv' % (folds_dir, fold),
                                      '%s/b_%d.csv' % (folds_dir, fold),
                                      transposed))
        self.X = np.vstack([f.X for f in folds])
        self.Y = np.vstack([f.Y for f in folds])
