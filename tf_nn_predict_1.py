#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf


def read_data(x_filename, y_filename):
    X = pd.read_csv(x_filename, header = None).as_matrix().T
    ideal_output_data = pd.read_csv(y_filename, header = None).as_matrix()
    Y = np.zeros([ideal_output_data.shape[1], 4], dtype=int)

    for i in xrange(ideal_output_data.shape[1]):
        for j in xrange(4):
             Y[i,j] = int(2 * ideal_output_data[2 * j, i] + ideal_output_data[2 * j + 1, i])
    return X, Y, ideal_output_data


def main(x_filename, y_filename, model_filename):
    x_ = tf.placeholder(dtype = tf.float32, shape = (None, 8))
    y_0_ = tf.placeholder(dtype = tf.float32, shape = (None, 4))
    y_1_ = tf.placeholder(dtype = tf.float32, shape = (None, 4))
    y_2_ = tf.placeholder(dtype = tf.float32, shape = (None, 4))
    y_3_ = tf.placeholder(dtype = tf.float32, shape = (None, 4))

    n_shared = 32
    shared_hidden_ = tf.contrib.layers.fully_connected(x_, n_shared, 
                                                      activation_fn=tf.nn.relu,
                                                      biases_initializer=tf.zeros_initializer())

    n_hidden = 16
    activation_hidden = tf.nn.relu
    inner_0_ = tf.contrib.layers.fully_connected(shared_hidden_, n_hidden, 
                                                 activation_fn=activation_hidden,
                                                 biases_initializer=tf.zeros_initializer())
    inner_1_ = tf.contrib.layers.fully_connected(shared_hidden_, n_hidden, 
                                                 activation_fn=activation_hidden,
                                                 biases_initializer=tf.zeros_initializer())
    inner_2_ = tf.contrib.layers.fully_connected(shared_hidden_, n_hidden, 
                                                 activation_fn=activation_hidden,
                                                 biases_initializer=tf.zeros_initializer())
    inner_3_ = tf.contrib.layers.fully_connected(shared_hidden_, n_hidden, 
                                                 activation_fn=activation_hidden,
                                                 biases_initializer=tf.zeros_initializer())

    n_output = 4
    activation_output = tf.nn.softmax
    output_0_ = tf.contrib.layers.fully_connected(inner_0_, n_output, 
                                                 activation_fn=activation_output,
                                                 biases_initializer=tf.zeros_initializer())
    output_1_ = tf.contrib.layers.fully_connected(inner_1_, n_output, 
                                                 activation_fn=activation_output,
                                                 biases_initializer=tf.zeros_initializer())
    output_2_ = tf.contrib.layers.fully_connected(inner_2_, n_output, 
                                                 activation_fn=activation_output,
                                                 biases_initializer=tf.zeros_initializer())
    output_3_ = tf.contrib.layers.fully_connected(inner_3_, n_output, 
                                                 activation_fn=activation_output,
                                                 biases_initializer=tf.zeros_initializer())

    loss_0_ = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_0_, logits = output_0_))
    loss_1_ = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_1_, logits = output_1_))
    loss_2_ = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_2_, logits = output_2_))
    loss_3_ = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_3_, logits = output_3_))

    general_loss_ = loss_0_ + loss_1_ + loss_2_ + loss_3_

    learning_rate = 0.001
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_ = optimizer.minimize(general_loss_)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess = tf.Session()

    sess.run(init)

    saver.restore(sess, model_filename)
    print 'Model loaded'

    print 'Loading test data'

    X_test_matrix, Y_test_matrix, ideal_output_test = read_data(x_filename, y_filename)
    
    Y_test_0 = sess.run(tf.one_hot(Y_test_matrix[:,0], depth=4))
    Y_test_1 = sess.run(tf.one_hot(Y_test_matrix[:,1], depth=4))
    Y_test_2 = sess.run(tf.one_hot(Y_test_matrix[:,2], depth=4))
    Y_test_3 = sess.run(tf.one_hot(Y_test_matrix[:,3], depth=4))
    print 'Evaluating model...'

    acc, errors, total_count = binary_accuracy([output_0_, output_1_, output_2_, output_3_], sess, 
        {x_: X_test_matrix, y_0_: Y_test_0, y_1_: Y_test_1, y_2_: Y_test_2, y_3_: Y_test_3},
        ideal_output_test)
    print 'Model binary accuracy: %.7f  %s errors of %s total' % (acc, errors, total_count)

    acc, errors, total_count = data_accuracy([output_0_, output_1_, output_2_, output_3_], sess, 
        {x_: X_test_matrix, y_0_: Y_test_0, y_1_: Y_test_1, y_2_: Y_test_2, y_3_: Y_test_3},
        Y_test_matrix)
    print 'Model data accuracy: %.7f  %s errors of %s total' % (acc, errors, total_count)


def convert_row_binary(x):
    i = x.argmax(axis=0)
    i0 = i / 2
    i1 = i % 2
    return [i0, i1]


def convert_row_data(x):
    i = x.argmax(axis=0)
    return [i]


def binary_accuracy(nn_out_layers, session, feed_dict, ideal_output_test):
    nn_outputs = session.run(nn_out_layers, feed_dict)

    predicted_y_matrix = np.hstack(map(
        lambda matrix: np.apply_along_axis(convert_row_binary, 1, matrix), 
        nn_outputs)).T
    total_elems = (ideal_output_test.shape[0] * ideal_output_test.shape[1])
    good = np.sum(predicted_y_matrix == ideal_output_test)
    return good / float(total_elems), total_elems - good, total_elems


def data_accuracy(nn_out_layers, session, feed_dict, Y_matrix):
    nn_outputs = session.run(nn_out_layers, feed_dict)
    predicted_y_matrix = np.hstack(map(
        lambda matrix: np.apply_along_axis(convert_row_data, 1, matrix), 
        nn_outputs))

    total_elems = (Y_matrix.shape[0] * Y_matrix.shape[1])
    good = np.sum(predicted_y_matrix.reshape((total_elems, )) == Y_matrix.reshape((total_elems, )))
    return good / float(total_elems), total_elems - good, total_elems

    # total_elems = (ideal_output_test.shape[0] * ideal_output_test.shape[1])
    # good = np.sum(predicted_y_matrix == ideal_output_test)
    # return good / float(total_elems), total_elems - good, total_elems


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('X', help = 'Data from output antennas (CSV)')
    parser.add_argument('Y', help = 'Data from input antennas (CSV)')
    parser.add_argument('model', help = 'Trained model path')
    args = parser.parse_args()
    main(args.X, args.Y, args.model)
