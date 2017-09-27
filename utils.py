import numpy as np


N_BITS = 8


def read_data(x_filename, y_filename, transposed=True):
    X = np.loadtxt(x_filename, delimiter=',', unpack=transposed)
    Y = np.loadtxt(y_filename, delimiter=',', unpack=transposed)
    return X, Y

def next_batch(X, Y, batch_size=100):
    perm = np.random.permutation(X.shape[0])[batch_size]
    return X[perm, :], Y[perm, :]
        