#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import json
import numpy as np
import pandas as pd
import tensorflow as tf


logger = logging.getLogger(__name__)


# model_filename = '/Users/mir/tf-models/mimo-1-3.ckpt'


def read_data(x_filename, y_filename):
    X = pd.read_csv(x_filename, header = None).as_matrix().T
    ideal_output_data = pd.read_csv(y_filename, header = None).as_matrix()
    Y = np.zeros([ideal_output_data.shape[1], 4], dtype=int)

    for i in xrange(ideal_output_data.shape[1]):
        for j in xrange(4):
             Y[i,j] = int(2 * ideal_output_data[2 * j, i] + ideal_output_data[2 * j + 1, i])
    return X, Y, ideal_output_data


def train_network(train_data, test_data, params, output_fd):
    X_matrix, Y_matrix, ideal_output_train = train_data
    X_test_matrix, Y_test_matrix, ideal_output_test = test_data
    
    sess = tf.Session()

    Y_0 = sess.run(tf.one_hot(Y_matrix[:,0], depth=4))
    Y_1 = sess.run(tf.one_hot(Y_matrix[:,1], depth=4))
    Y_2 = sess.run(tf.one_hot(Y_matrix[:,2], depth=4))
    Y_3 = sess.run(tf.one_hot(Y_matrix[:,3], depth=4))
    
    Y_test_0 = sess.run(tf.one_hot(Y_test_matrix[:,0], depth=4))
    Y_test_1 = sess.run(tf.one_hot(Y_test_matrix[:,1], depth=4))
    Y_test_2 = sess.run(tf.one_hot(Y_test_matrix[:,2], depth=4))
    Y_test_3 = sess.run(tf.one_hot(Y_test_matrix[:,3], depth=4))

    logger.info('Creating model...')

    x_ = tf.placeholder(dtype = tf.float32, shape = (None, 8))
    y_0_ = tf.placeholder(dtype = tf.float32, shape = (None, 4))
    y_1_ = tf.placeholder(dtype = tf.float32, shape = (None, 4))
    y_2_ = tf.placeholder(dtype = tf.float32, shape = (None, 4))
    y_3_ = tf.placeholder(dtype = tf.float32, shape = (None, 4))

    n_shared = params.get('n_shared', 32)
    shared_hidden_ = tf.contrib.layers.fully_connected(x_, n_shared, 
                                                      activation_fn=tf.nn.relu,
                                                      biases_initializer=tf.zeros_initializer())

    n_hidden = params.get('n_hidden', 16)
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

    learning_rate = params.get('learning_rate', 0.01)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_ = optimizer.minimize(general_loss_)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess = tf.Session()

    logger.info('Initialize model...')
    sess.run(init)

    logger.info('Training...')
    max_iterations = params.get('max_iterations', 10000)
    batch_size = params.get('batch_size', 300)
    ce_threshold = params.get('ce_threshold', 1e-4)

    prev_train_ce = None
    for i in xrange(max_iterations):
        perm = np.random.permutation(X_matrix.shape[0])
        X_matrix_p = X_matrix[perm, :]
        Y_0_p = Y_0[perm, :]
        Y_1_p = Y_1[perm, :]
        Y_2_p = Y_2[perm, :]
        Y_3_p = Y_3[perm, :]
        for b in xrange(X_matrix.shape[0] / batch_size):
            batch_x = X_matrix_p[batch_size * b:batch_size * (b + 1),:]
            batch_y_0 = Y_0_p[batch_size * b:batch_size * (b + 1),:]
            batch_y_1 = Y_1_p[batch_size * b:batch_size * (b + 1),:]
            batch_y_2 = Y_2_p[batch_size * b:batch_size * (b + 1),:]
            batch_y_3 = Y_3_p[batch_size * b:batch_size * (b + 1),:]
            feed_dict={x_: batch_x, 
                       y_0_: batch_y_0, 
                       y_1_: batch_y_1, 
                       y_2_: batch_y_2, 
                       y_3_: batch_y_3}
            sess.run(train_, feed_dict=feed_dict)
        if i % 10 == 0:
            train_feed_dict = {x_: X_matrix, y_0_: Y_0, y_1_: Y_1, y_2_: Y_2, y_3_: Y_3}
            train_ce = sess.run(general_loss_, train_feed_dict)
            train_ber = bit_error_rate([output_0_, output_1_, output_2_, output_3_], sess, train_feed_dict, ideal_output_train)
            logger.debug('Step: %d train CE: %.5f train BER: %.5f', i, train_ce, train_ber)
            if prev_train_ce is not None and np.abs(prev_train_ce - train_ce) < ce_threshold:
                break
            prev_train_ce = train_ce
    logger.info('Finished training...')
    train_feed_dict = {x_: X_matrix, y_0_: Y_0, y_1_: Y_1, y_2_: Y_2, y_3_: Y_3}
    train_ce = sess.run(general_loss_, train_feed_dict)
    train_ber = bit_error_rate([output_0_, output_1_, output_2_, output_3_], sess, train_feed_dict, ideal_output_train)
    test_feed_dict = {x_: X_test_matrix, y_0_: Y_test_0, y_1_: Y_test_1, y_2_: Y_test_2, y_3_: Y_test_3}
    test_ce = sess.run(general_loss_, test_feed_dict)
    test_ber = bit_error_rate([output_0_, output_1_, output_2_, output_3_], sess, test_feed_dict, ideal_output_test)
    logger.info('Params: %s max step: %d train CE: %.5f train BER: %.5f test CE: %.5f test BER: %.5f' % (json.dumps(params), i, train_ce, train_ber, test_ce, test_ber))
    print >> output_fd, '%s\t%d\t%.5f\t%.5f\t%.5f\t%.5f' % (json.dumps(params), i, train_ce, train_ber, test_ce, test_ber)
    logger.info('Done for params %s', params)
    output_fd.flush()


def convert_row(x):
    i = x.argmax(axis=0)
    i0 = i / 2
    i1 = i % 2
    return [i0, i1]


def bit_error_rate(nn_out_layers, session, feed_dict, ideal_output_test):
    nn_outputs = session.run(nn_out_layers, feed_dict)

    predicted_y_matrix = np.hstack(map(
        lambda matrix: np.apply_along_axis(convert_row, 1, matrix), 
        nn_outputs)).T
    total_elems = (ideal_output_test.shape[0] * ideal_output_test.shape[1])
    return np.sum(predicted_y_matrix != ideal_output_test) / float(total_elems)


def main():
    logger.info('Reading train data...')
    X_matrix, Y_matrix, ideal_output_train = read_data('data/1/y.csv', 'data/1/b.csv')

    logger.info('Reading test data...')
    # X_test_matrix, Y_test_matrix, ideal_output_test = read_data('data/1/ytest.csv', 'data/1/btest.csv')
    X_test_matrix, Y_test_matrix, ideal_output_test = read_data('~/y.csv1', '~/b.csv1')
    
    logger.info('Data is ready!')

    with open('train_results.tsv', 'w') as output_fd:
        for n_shared_exponent in xrange(3, 10):
            for n_hidden_exponent in xrange(2, 7):
                params = {'n_shared': 2 ** n_shared_exponent, 'n_hidden': 2 ** n_hidden_exponent}
                logger.debug('Training for params: %s', params)
                train_network((X_matrix, Y_matrix, ideal_output_train), (X_test_matrix, Y_test_matrix, ideal_output_test), params, output_fd)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s')
    logger.info('Start')
    main()
