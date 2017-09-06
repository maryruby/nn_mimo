#!/usr/bin/env python
# -*- coding: utf-8 -*-

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


def main():
    # saver = tf.train.Saver()
    sess = tf.Session()

    print 'Reading train data...'
    X_matrix, Y_matrix, ideal_output_train = read_data('data/1/y.csv', 'data/1/b.csv')

    Y_0 = sess.run(tf.one_hot(Y_matrix[:,0], depth=4))
    Y_1 = sess.run(tf.one_hot(Y_matrix[:,1], depth=4))
    Y_2 = sess.run(tf.one_hot(Y_matrix[:,2], depth=4))
    Y_3 = sess.run(tf.one_hot(Y_matrix[:,3], depth=4))

    print 'Reading test data...'
    X_test_matrix, Y_test_matrix, ideal_output_test = read_data('data/1/ytest.csv', 'data/1/btest.csv')
    
    Y_test_0 = sess.run(tf.one_hot(Y_test_matrix[:,0], depth=4))
    Y_test_1 = sess.run(tf.one_hot(Y_test_matrix[:,1], depth=4))
    Y_test_2 = sess.run(tf.one_hot(Y_test_matrix[:,2], depth=4))
    Y_test_3 = sess.run(tf.one_hot(Y_test_matrix[:,3], depth=4))

    print 'Creating model...'


    x_ = tf.placeholder(dtype = tf.float32, shape = (None, 8))
    y_0_ = tf.placeholder(dtype = tf.float32, shape = (None, 4))
    y_1_ = tf.placeholder(dtype = tf.float32, shape = (None, 4))
    y_2_ = tf.placeholder(dtype = tf.float32, shape = (None, 4))
    y_3_ = tf.placeholder(dtype = tf.float32, shape = (None, 4))

    n_shared = 32
    shared_hidden_ = tf.contrib.layers.fully_connected(x_, n_shared, 
                                                      activation_fn=tf.nn.relu,
                                                      biases_initializer=tf.zeros_initializer())

    n_hidden = 4
    activation_hidden = tf.nn.softmax
    hidden_0_ = tf.contrib.layers.fully_connected(shared_hidden_, n_hidden, 
                                                 activation_fn=activation_hidden,
                                                 biases_initializer=tf.zeros_initializer())
    hidden_1_ = tf.contrib.layers.fully_connected(shared_hidden_, n_hidden, 
                                                 activation_fn=activation_hidden,
                                                 biases_initializer=tf.zeros_initializer())
    hidden_2_ = tf.contrib.layers.fully_connected(shared_hidden_, n_hidden, 
                                                 activation_fn=activation_hidden,
                                                 biases_initializer=tf.zeros_initializer())
    hidden_3_ = tf.contrib.layers.fully_connected(shared_hidden_, n_hidden, 
                                                 activation_fn=activation_hidden,
                                                 biases_initializer=tf.zeros_initializer())

    loss_0_ = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_0_, logits = hidden_0_))
    loss_1_ = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_1_, logits = hidden_1_))
    loss_2_ = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_2_, logits = hidden_2_))
    loss_3_ = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_3_, logits = hidden_3_))

    general_loss_ = loss_0_ + loss_1_ + loss_2_ + loss_3_

    learning_rate = 0.001
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_ = optimizer.minimize(general_loss_)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess = tf.Session()

    print 'Initialize model...'
    sess.run(init)

    print 'Training...'
    # max_iterations = 13200
    max_iterations = 300  # дальше мы не обучились
    batch_size = 100
    for i in xrange(max_iterations):
        for b in xrange(X_matrix.shape[0] / batch_size):
            batch_x = X_matrix[batch_size * b:batch_size * (b + 1),:]
            batch_y_0 = Y_0[batch_size * b:batch_size * (b + 1),:]
            batch_y_1 = Y_1[batch_size * b:batch_size * (b + 1),:]
            batch_y_2 = Y_2[batch_size * b:batch_size * (b + 1),:]
            batch_y_3 = Y_3[batch_size * b:batch_size * (b + 1),:]
            feed_dict={x_: batch_x, 
                       y_0_: batch_y_0, 
                       y_1_: batch_y_1, 
                       y_2_: batch_y_2, 
                       y_3_: batch_y_3}
            sess.run(train_, feed_dict=feed_dict)
        if i % 10 == 0:
            test_feed_dict = {x_: X_test_matrix, y_0_: Y_test_0, y_1_: Y_test_1, y_2_: Y_test_2, y_3_: Y_test_3}
            ce = sess.run(general_loss_, test_feed_dict)
            acc = accuracy([hidden_0_, hidden_1_, hidden_2_, hidden_3_], sess, test_feed_dict,ideal_output_test)
            print 'Step: %d CE: %.5f ACC: %.5f' % (i, ce, acc)
                    
        if i % 100 == 0 and i > 0:
            print 'Saving model at step %d' % i
            save_path = saver.save(sess, '/Users/mir/tf-models/mimo-1-2.ckpt', global_step = i)

    print 'Done training!'

    acc = accuracy([hidden_0_, hidden_1_, hidden_2_, hidden_3_], sess,
        {x_: X_test_matrix, y_0_: Y_test_0, y_1_: Y_test_1, y_2_: Y_test_2, y_3_: Y_test_3},
        ideal_output_test)
    print 'Model final accuracy: %.5f' % acc

    save_path = saver.save(sess, '/Users/mir/tf-models/mimo-1-2.ckpt', global_step = max_iterations)
    print 'Model stored in %s' % save_path


def convert_row(x):
    i = x.argmax(axis=0)
    i0 = i / 2
    i1 = i % 2
    return [i0, i1]


def accuracy(nn_out_layers, session, feed_dict, ideal_output_test):
    nn_outputs = session.run(nn_out_layers, feed_dict)

    predicted_y_matrix = np.hstack(map(
        lambda matrix: np.apply_along_axis(convert_row, 1, matrix), 
        nn_outputs)).T
    total_elems = (ideal_output_test.shape[0] * ideal_output_test.shape[1])
    return np.sum(predicted_y_matrix == ideal_output_test) / float(total_elems)


if __name__ == '__main__':
    print 'Start'
    main()
