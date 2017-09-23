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


def main(x_filename, y_filename, model_filename, output_filename):
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

    X_matrix, Y_matrix, _ = read_data(x_filename, y_filename)
    
    Y_0 = sess.run(tf.one_hot(Y_matrix[:,0], depth=4))
    Y_1 = sess.run(tf.one_hot(Y_matrix[:,1], depth=4))
    Y_2 = sess.run(tf.one_hot(Y_matrix[:,2], depth=4))
    Y_3 = sess.run(tf.one_hot(Y_matrix[:,3], depth=4))
    print 'Evaluating model...'

    predicted_y_matrix = make_predictions([output_0_, output_1_, output_2_, output_3_], sess, 
        {x_: X_matrix, y_0_: Y_0, y_1_: Y_1, y_2_: Y_2, y_3_: Y_3})
    dump_to_file(X_matrix, Y_matrix, predicted_y_matrix, output_filename)
    print 'Done!'


def convert_row_data(x):
    i = x.argmax(axis=0)
    return [i]


def dump_to_file(X_matrix, Y_ideal, Y_predicted, output_filename):
    flat = []
    for i in range(Y_ideal.shape[1]):
        part_Y_ideal = Y_ideal[:, i]
        part_Y_predicted = Y_predicted[:, i]
        part_X_0 = X_matrix[:, 2 * i]
        part_X_1 = X_matrix[:, 2 * i + 1]
        flat.append(np.vstack([part_Y_ideal, part_Y_predicted, part_X_0, part_X_1]))
    flat_matrix = np.hstack(flat)
    np.savetxt(fname=output_filename, X=flat_matrix.T, delimiter = ' ', newline='\n', fmt='%.4f')


def make_predictions(nn_out_layers, session, feed_dict):
    nn_outputs = session.run(nn_out_layers, feed_dict)
    predicted_y_matrix = np.hstack(map(
        lambda matrix: np.apply_along_axis(convert_row_data, 1, matrix), 
        nn_outputs))

    return predicted_y_matrix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('X', help = 'Data from output antennas (CSV), usually called y.csv')
    parser.add_argument('Y', help = 'Data from input antennas (CSV), usually called b.csv')
    parser.add_argument('model', help = 'Trained model path')
    parser.add_argument('output', help = 'Output file for predictions')
    args = parser.parse_args()
    main(args.X, args.Y, args.model, args.output)
