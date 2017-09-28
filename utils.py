import numpy as np


N_BITS = 8


def read_data(x_filename, y_filename, transposed=True):
    X = np.loadtxt(x_filename, delimiter=',', unpack=transposed)
    Y = np.loadtxt(y_filename, delimiter=',', unpack=transposed)
    return X, Y


def generate_batches(X, Y, batch_size=100):
    perm = np.random.permutation(X.shape[0])
    X_p = X[perm, :]
    Y_p = Y[perm, :]
    total_batches = X.shape[0] / batch_size
    for b in xrange(total_batches):
        batch_x = X_p[batch_size * b:batch_size * (b + 1), :]
        batch_y = Y_p[batch_size * b:batch_size * (b + 1), :]
        yield batch_x, batch_y
