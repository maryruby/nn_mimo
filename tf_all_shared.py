#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from functools import partial
import argparse

import quality


logger = logging.getLogger(__name__)


model_filename_template = 'models/mimo-all-shared-%d-%d'


def read_data(x_filename, y_filename):
    X = pd.read_csv(x_filename, header = None).as_matrix().T
    Y = pd.read_csv(y_filename, header = None).as_matrix().T
    
    return X, Y


def main(n_shared, n_hidden, max_iterations, lr, eps):
    sess = tf.Session()

    logger.info('Reading train data...')
    X_matrix, Y_matrix = read_data('data/1/y_10_7.csv', 'data/1/b_10_7.csv')
    
    logger.info('Reading validation data...')
    X_val_matrix, Y_val_matrix = read_data('data/1/y.csv', 'data/1/b.csv')
    
    logger.info('Creating model...')

    n_bits = 8
    x_ = tf.placeholder(dtype = tf.float32, shape = (None, n_bits))
    y_ = tf.placeholder(dtype = tf.float32, shape = (None, n_bits))

    hiddens = [tf.contrib.layers.fully_connected(x_, n_shared, 
                                                   activation_fn=tf.nn.relu,
                                                   biases_initializer=tf.zeros_initializer())]
    for i in xrange(n_hidden - 1):
        hiddens.append(tf.contrib.layers.fully_connected(hiddens[-1], n_shared, 
                                                      activation_fn=tf.nn.relu,
                                                      biases_initializer=tf.zeros_initializer()))
    output = tf.contrib.layers.fully_connected(hiddens[-1], n_bits, 
                                               activation_fn=tf.nn.sigmoid,
                                               biases_initializer=tf.zeros_initializer())
    general_loss_ = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=output))

    optimizer = tf.train.GradientDescentOptimizer(lr)
    train_ = optimizer.minimize(general_loss_)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess = tf.Session()

    logger.info('Initialize model...')
    sess.run(init)

    logger.info('Restore model...')
    min_iterations = 0
    # saver.restore(sess, '%s-%s' % (model_filename, min_iterations))

    logger.info('Training...')
    batch_size = 300
    threshold = 0.5
    prev_ce = None
    for i in xrange(min_iterations, max_iterations):
        perm = np.random.permutation(X_matrix.shape[0])
        X_matrix_p = X_matrix[perm, :]
        Y_p = Y_matrix[perm, :]
        
        for b in xrange(X_matrix.shape[0] / batch_size):
            batch_x = X_matrix_p[batch_size * b:batch_size * (b + 1),:]
            batch_y = Y_p[batch_size * b:batch_size * (b + 1),:]
            sess.run(train_, feed_dict={x_: batch_x, y_: batch_y})
        if i % 10 == 0:
            train_ce, train_cber, train_ber, train_rer = metrics(x_, y_, X_matrix, Y_matrix, loss, nn_out_layer, sess)
            logger.info('TRAIN step: %d column BER: [%s]', i, ','.join('%.5f' % c for c in train_cber))
            logger.info('TRAIN step: %d CE: %.5f BER: %.5f RER: %.5f', i, train_ce, train_ber, train_rer)

            val_ce, val_cber, val_ber, val_rer = metrics(x_, y_, X_val_matrix, Y_val_matrix, loss, nn_out_layer, sess)
            logger.info('VALIDATION step: %d column BER: [%s]', i, ','.join('%.5f' % c for c in val_cber))
            logger.info('VALIDATION step: %d CE: %.5f BER: %.5f RER: %.5f', i, val_ce, val_ber, val_rer)
            if prev_ce and abs(train_ce - prev_ce) < eps:
                logger.info('Converged at %d' % i)
                max_iterations = i
                break
            prev_ce = train_ce
                    
        if i % 100 == 0 and i > min_iterations:
            logger.info('Saving model at step %d .......' % i,)
            save_path = saver.save(sess, model_filename_template % (n_shared, n_hidden), global_step = i)
            logger.info('ok')

    logger.info('Done training!')

    logger.info('Reading final test data...')
    X_test_matrix, Y_test_matrix = read_data('data/1/y_big_test.csv', 'data/1/b_big_test.csv')
    test_ce, test_cber, test_ber, test_rer = metrics(x_, y_, X_test_matrix, Y_test_matrix, loss, nn_out_layer, sess)
    logger.info('TEST column BER: [%s]', ','.join('%.5f' % c for c in test_cber))
    logger.info('TEST CE: %.5f BER: %.5f RER: %.5f', test_ce, test_ber, test_rer)

    save_path = saver.save(sess, model_filename_template % (n_shared, n_hidden), global_step = max_iterations)
    logger.info('Model stored in %s' % save_path)


def make_predicted_matrix(nn_output_layer, session, feed_dict, threshold=0.5):
    nn_output = session.run(nn_out_layer, feed_dict)
    return (nn_output >= threshold).astype(int)


def metrics(x_, y_, X, Y, loss, out_layer, sess, threshold=0.5):
    feed_dict = {x_: X, y_: Y}
    ce = sess.run(loss, feed_dict)
    predicted = make_predicted_matrix(out_layer, sess, feed_dict, threshold)
    cber = quality.column_bit_error_rate(predicted, Y)
    ber = np.mean(cber)
    rer = quality.row_error_rate(predicted, Y)
    return ce, cber, ber, rer


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden', help='Size of hidden layer', type=int, required=True)
    parser.add_argument('--layers', help='Total hidden layers', type=int, required=True)
    parser.add_argument('--max-iterations', help='Max iterations', type=int, default=10000)
    parser.add_argument('--lr', help='Learning rate', type=float, default=0.01)
    parser.add_argument('--eps', help='Convergence CE diff', type=float, default=1e-6)
    args = parser.parse_args()
    logger.info('Start')
    main(args.hidden, args.layers, args.max_iterations, args.lr, args.eps)
