#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pandas as pd
import tensorflow as tf


def read_data(x_filename, y_filename, transposed):
    X = pd.read_csv(x_filename, header = None).as_matrix()
    Y = pd.read_csv(y_filename, header = None).as_matrix()
    if transposed:
        return X.T, Y.T
    else:
        return X, Y


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def convert_to_tfrecords(X, Y, filename):
    if X.shape[0] != Y.shape[0]:
        raise ValueError('Received examples count %d does not match send examples count %d' % (X.shape[0], Y.shape[0]))
    num_examples = X.shape[0]
    writer = tf.python_io.TFRecordWriter(filename)
    for i in xrange(num_examples):
        received = X[i, :]
        sent = Y[i, :]
        example = tf.train.Example(features=tf.train.Features(feature={
            'received': float_list_feature(received),
            'sent': float_list_feature(sent)
            }))
        writer.write(example.SerializeToString())
    writer.close()


def main(X_filename, Y_filename, output_filename, transposed):
    X, Y = read_data(X_filename, Y_filename, transposed)
    convert_to_tfrecords(X, Y, output_filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--received', help='CSV with received signals matrix (N_examples rows, 2*K_antennas columns)')
    parser.add_argument('--sent', help='CSV with sent signals matrix (N_examples rows, 2*K_antennas columns)')
    parser.add_argument('--filename', help='Output filename')
    parser.add_argument('--transposed', help='Specify if input data is transposed', action='store_true', default=False)
    args = parser.parse_args()
    main(args.received, args.sent, args.filename, args.transposed)
