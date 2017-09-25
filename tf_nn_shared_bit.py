#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from functools import partial


logger = logging.getLogger(__name__)


model_filename = 'models/mimo-2-1'


def read_data(x_filename, y_filename):
    X = pd.read_csv(x_filename, header = None).as_matrix().T
    Y = pd.read_csv(y_filename, header = None).as_matrix().T
    
    return X, Y


def main():
    # saver = tf.train.Saver()
    sess = tf.Session()

    logger.info('Reading train data...')
    X_matrix, Y_matrix = read_data('data/1/y.csv', 'data/1/b.csv')

    
    logger.info('Reading test data...')
    X_test_matrix, Y_test_matrix = read_data('data/1/y_big_test.csv', 'data/1/b_big_test.csv')
    
    logger.info('Creating model...')

    n_bits = 8
    x_ = tf.placeholder(dtype = tf.float32, shape = (None, n_bits))
    y_ = [tf.placeholder(dtype = tf.float32, shape = (None, 1)) for _ in range(n_bits)]

    n_shared = 32
    shared_hidden_ = tf.contrib.layers.fully_connected(x_, n_shared, 
                                                      activation_fn=tf.nn.relu,
                                                      biases_initializer=tf.zeros_initializer())

    n_hidden = 16
    activation_hidden = tf.nn.relu
    inners = []
    for i in range(n_bits):
        inners.append(tf.contrib.layers.fully_connected(shared_hidden_, n_hidden, 
                                                 activation_fn=activation_hidden,
                                                 biases_initializer=tf.zeros_initializer()))
    outputs = []
    loss = []
    n_output = 1
    activation_output = tf.nn.sigmoid
    for i in range(n_bits):
        outputs.append(tf.contrib.layers.fully_connected(inners[i], n_output, 
                                                activation_fn=activation_output,
                                                biases_initializer=tf.zeros_initializer()))
        loss.append(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=y_[i],
            logits=outputs[i])))
        

    general_loss_ = tf.reduce_sum(loss)

    learning_rate = 0.01
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
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
    # max_iterations = 13200
    max_iterations = 10000
    batch_size = 100
    treshold = 0.5
    for i in xrange(min_iterations, max_iterations):
        perm = np.random.permutation(X_matrix.shape[0])
        X_matrix_p = X_matrix[perm, :]
        Y_p = Y_matrix[perm, :]
        
        for b in xrange(X_matrix.shape[0] / batch_size):
            batch_x = X_matrix_p[batch_size * b:batch_size * (b + 1),:]
            batch_y = Y_p[batch_size * b:batch_size * (b + 1),:]
            feed_dict = make_feed_dict(x_, y_, batch_x, batch_y)
            sess.run(train_, feed_dict=feed_dict)
        if i % 10 == 0:
            train_feed_dict = make_feed_dict(x_, y_, X_matrix, Y_matrix)
            train_ce = sess.run(general_loss_, train_feed_dict)
            train_rer = row_error_rate(outputs, sess, train_feed_dict, Y_matrix, treshold)

            test_feed_dict = make_feed_dict(x_, y_, X_test_matrix, Y_test_matrix)
            test_ce = sess.run(general_loss_, test_feed_dict)
            test_rer = row_error_rate(outputs, sess, test_feed_dict, Y_test_matrix, treshold)
            logger.info('Step: %d train CE: %.5f train RER: %.5f test CE: %.5f test RER: %.5f' % (i, train_ce, train_rer, test_ce, test_rer))
                    
        if i % 100 == 0 and i > min_iterations:
            logger.info('Saving model at step %d .......' % i,)
            save_path = saver.save(sess, model_filename, global_step = i)
            logger.info('ok')

    logger.info('Done training!')

    test_feed_dict = make_feed_dict(x_, y_, X_test_matrix, Y_test_matrix)
    rer = row_error_rate(outputs, sess, test_feed_dict, Y_test_matrix, treshold)
    logger.info('Model final RER: %.5f' % rer)

    save_path = saver.save(sess, model_filename, global_step = max_iterations)
    logger.info('Model stored in %s' % save_path)


def make_feed_dict(x_, y_, X, Y):
    feed_dict = {x_: X}
    for j in range(len(y_)): 
        feed_dict[y_[j]] = Y[:,[j]]#.reshape(Y.shape[0], 1)
    return feed_dict


def convert_row(x, treshold):
    return (x>= treshold).astype(int)


def row_error_rate(nn_out_layers, session, feed_dict, Y_matrix, treshold=0.5):
    nn_outputs = session.run(nn_out_layers, feed_dict)

    predicted_y_matrix = np.hstack(map(
        lambda matrix: np.apply_along_axis(partial(convert_row, treshold = treshold), 1, matrix), 
        nn_outputs))
    total_elems = (Y_matrix.shape[0] * Y_matrix.shape[1])
    return 1.0 - np.sum(np.all(np.equal(predicted_y_matrix, Y_matrix), 1)) / float(total_elems)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s')
    logger.info('Start')
    main()
