import numpy as np
import random
import utils

N_BITS = utils.N_BITS


class DataSet(object):

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def batches_generator(batch_size=100):
        for b_x, b_y in utils.generate_batches(self.X, self.Y, batch_size):
            yield DataSet(b_x, b_y)


def read_dataset(x_filename, y_filename, transposed=True):
    X = np.loadtxt(x_filename, delimiter=',', unpack=transposed)
    Y = np.loadtxt(y_filename, delimiter=',', unpack=transposed)
    return DataSet(X, Y)


class FoldedDataSet(object):
    def __init__(self, folds_dir, n_folds):
        self.folds = []
        for fold in xrange(n_folds):
            self.folds.append(read_dataset('%s/y_%d.csv' % (folds_dir, fold), '%s/b_%d.csv' % (folds_dir, fold)))


    def batches_generator(batch_size=100):
        fold_generators = map(lambda fold: fold.batches_generator(batch_size), self.folds)

        while len(fold_generators) > 0:
            random_fold = random.randint(len(fold_generators))
            try:
                yield fold_generators[random_fold].next()
            except StopIteration:
                del fold_generators[random_fold]



