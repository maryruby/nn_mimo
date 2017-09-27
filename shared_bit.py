#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import logging
import numpy as np
import tensorflow as tf
import time

import utils
import quality


logger = logging.getLogger(__name__)



def create_model(x, args):
    shared_hidden = tf.contrib.layers.fully_connected(x, args.n_shared,
                                                      activation_fn=tf.nn.relu,
                                                      biases_initializer=tf.zeros_initializer())
    outputs = []
    for i in range(utils.N_BITS):
        inner = tf.contrib.layers.fully_connected(shared_hidden, args.n_hidden,
                                                  activation_fn=tf.nn.relu,
                                                  biases_initializer=tf.zeros_initializer())
        outputs.append(tf.contrib.layers.fully_connected(inner, 1,
                                                         activation_fn=tf.nn.sigmoid,
                                                         biases_initializer=tf.zeros_initializer()))
    # after stack we will get shape (?, N_BITS, 1), then we squeeze it to (?, N_BITS)
    return tf.squeeze(tf.stack(outputs, axis=1))


def create_loss(logits, labels):
    return tf.reduce_mean(tf.reduce_sum(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits), 
        axis=1))


def training(loss, learning_rate):
    # Add a scalar summary for the snapshot loss
    tf.summary.scalar('loss', loss)
    # Create the gradient descent optimizer with the given learning rate
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # Create a variable to track the global step
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def main(args):
    sess = tf.Session()

    logger.info('Creating model...')
    x_ = tf.placeholder(dtype = tf.float32, shape = (None, utils.N_BITS))
    y_ = tf.placeholder(dtype = tf.float32, shape = (None, utils.N_BITS))

    logits = create_model(x_, args)
    loss = create_loss(logits, y_)
    train_op = training(loss, args.learning_rate)

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    saver = tf.train.Saver()
    sess = tf.Session()

    logger.info('Reading train data...')
    X_train, Y_train = utils.read_data('data/1/y_big_test.csv', 'data/1/b_big_test.csv')
    
    logger.info('Reading validation data...')
    X_valid, Y_valid = utils.read_data('data/1/y.csv', 'data/1/b.csv')

    logger.info('Initialize model...')
    sess.run(init_op)

    logger.info('Training...')
    min_iterations = 0
    max_iterations = args.max_iterations
    batch_size = args.batch_size
    treshold = 0.5
    for epoch in xrange(min_iterations, max_iterations):
        start_time = time.time()
        perm = np.random.permutation(X_train.shape[0])
        X_p = X_train[perm, :]
        Y_p = Y_train[perm, :]

        for b in xrange(X_train.shape[0] / batch_size):
            batch_x = X_p[batch_size * b:batch_size * (b + 1),:]
            batch_y = Y_p[batch_size * b:batch_size * (b + 1),:]
            feed_dict = {x_: batch_x, y_: batch_y}
            sess.run(train_op, feed_dict=feed_dict)
        duration = time.time() - start_time
        logger.debug('Epoch %d done for %.2f secs', epoch, duration)

        if epoch % 10 == 0:
            train_predictions, train_ce = sess.run([logits, loss], {x_: X_train, y_: Y_train})
            train_predictions = (train_predictions >= treshold).astype(float)
            train_cber = quality.column_bit_error_rate(train_predictions, Y_train)
            train_ber = np.mean(train_cber)
            logger.info('TRAIN epoch: %d CE: %.5f column BER: [%s] (mean: %.5f)', 
                        epoch, train_ce, ','.join('%.5f' % c for c in train_cber), train_ber)
        
            val_predictions, val_ce = sess.run([logits, loss], {x_: X_valid, y_: Y_valid})
            val_predictions = (val_predictions >= treshold).astype(float)
            val_cber = quality.column_bit_error_rate(val_predictions, Y_valid)
            val_ber = np.mean(val_cber)
            logger.info('VALIDATION epoch: %d CE: %.5f column BER: [%s] (mean: %.5f)', 
                        epoch, val_ce, ','.join('%.5f' % c for c in val_cber), val_ber)
            
        if epoch % 100 == 0 and epoch > min_iterations:
            logger.info('Saving model at step %d .......', epoch)
            save_path = saver.save(sess, args.model_filename, global_step = epoch)
            logger.info('ok')

    logger.info('Done training!')

    logger.info('Reading final test data...')
    X_test, Y_test = utils.read_data('data/1/y_big_test.csv', 'data/1/b_big_test.csv')

    test_predictions = sess.run(logits, {x_: X_test, y_: Y_test})
    test_predictions = (test_predictions >= treshold).astype(float)
    test_cber = quality.column_bit_error_rate(test_predictions, Y_test)
    test_ber = np.mean(test_cber)
    logger.info('TEST column BER: [%s] (mean: %.5f)', ','.join('%.5f' % c for c in test_cber), test_ber)

    save_path = saver.save(sess, args.model_filename, global_step = max_iterations)
    logger.info('Model stored in %s' % save_path)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', help='Initial learning rate', type=float, default=0.01)
    parser.add_argument('--max_iterations', help='Number of epochs to run trainer', type=int, default=100)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=100)
    # parser.add_argument('--train', help='Directory with the training data')
    parser.add_argument('--n-shared', help='Size of shared hidden layer', type=int, default=32)
    parser.add_argument('--n-hidden', help='Size of separate hidden layer', type=int, default=16)
    parser.add_argument('--model-filename', help='Path to save model')
    args = parser.parse_args()
    main(args)
