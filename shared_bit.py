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
    with tf.name_scope('shared_hidden'):
        shared_hidden = tf.contrib.layers.fully_connected(x, args.n_shared,
                                                          activation_fn=tf.nn.relu,
                                                          biases_initializer=tf.zeros_initializer())
    with tf.name_scope('fingers'):
        outputs = []
        for i in range(utils.N_BITS):
            with tf.name_scope('hidden_%d' % i):
                inner = tf.contrib.layers.fully_connected(shared_hidden, args.n_hidden,
                                                          activation_fn=tf.nn.relu,
                                                          biases_initializer=tf.zeros_initializer())
            with tf.name_scope('output_%d' % i):
                outputs.append(tf.contrib.layers.fully_connected(inner, 1,
                                                                 activation_fn=tf.nn.sigmoid,
                                                                 biases_initializer=tf.zeros_initializer()))
                tf.summary.histogram('activations', outputs[-1])
    with tf.name_scope('activations'):
        # after stack we will get shape (?, N_BITS, 1), then we squeeze it to (?, N_BITS)
        predictions = tf.squeeze(tf.stack(outputs, axis=1))
        tf.summary.histogram('activations', predictions)
    return predictions


# logits = predicted tensor, labels = ideal tensor;
def create_loss(logits, labels):
    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(
            tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits), axis=1))
    tf.summary.scalar('cross_entropy', cross_entropy)
    return cross_entropy


def binary_error_rate(logits, labels, threshold=0.5):
    t = tf.constant(threshold)
    with tf.name_scope('ber'):
        with tf.name_scope('errors'):
            errors = tf.not_equal(tf.greater_equal(logits, t), labels)
        with tf.name_scope('column_ber'):
            column_ber = tf.reduce_mean(tf.cast(errors, tf.float32), axis=0)
            for column in xrange(utils.N_BITS):
                tf.summary.scalar('ber_%d' % column, column_ber[column])
    ber = tf.reduce_mean(column_ber)
    tf.summary.scalar('ber', ber)
    return column_ber, ber


def training(loss, args):
    with tf.name_scope('train'):
        # Create the gradient descent optimizer with the given learning rate
        optimizer = tf.train.GradientDescentOptimizer(args.learning_rate)
        # Create a variable to track the global step
        global_step = tf.Variable(0, name='global_step', trainable=False)
        tf.summary.scalar('global_step', global_step)
        # Use the optimizer to apply the gradients that minimize the loss
        # (and also increment the global step counter) as a single training step
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op


def main(args):
    sess = tf.Session()

    logger.info('Creating model...')
    with tf.name_scope('input'):
        x_ = tf.placeholder(dtype = tf.float32, shape = (None, utils.N_BITS), name='x-input')
        y_ = tf.placeholder(dtype = tf.float32, shape = (None, utils.N_BITS), name='y-input')

    logits = create_model(x_, args)
    loss = create_loss(logits, y_)
    cber, ber = binary_error_rate(logits, y_)
    train_op = training(loss, args)

    # Here is what my boss think I do:

    # for i in xrange(max_iter):
    #     data_x, data_y = next_train_batch()
    #     feed_dict = {x_: data_x, y_: data_y}
    #     _, ce = sess.run([train_op, loss], feed_dict)
    # data_x, data_y = get_test_data()
    # feed_dict = {x_: data_x, y_: data_y}
    # metrics = sess.run([cber, ber], feed_dict)


    # Here is what I actually do:

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    saver = tf.train.Saver()
    merged = tf.summary.merge_all()
    if tf.gfile.Exists(args.log_dir):
        tf.gfile.DeleteRecursively(args.log_dir)
    tf.gfile.MakeDirs(args.log_dir)
    train_writer = tf.summary.FileWriter(args.log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(args.log_dir + '/test')

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
    threshold = 0.5
    for epoch in xrange(min_iterations, max_iterations):
        start_time = time.time()
        perm = np.random.permutation(X_train.shape[0])
        X_p = X_train[perm, :]
        Y_p = Y_train[perm, :]

        if epoch % 10 == 0:
            ## unknown functions
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
        else:
            run_options = None
            run_metadata = None
        total_batches = X_train.shape[0] / batch_size
        for b in xrange(total_batches):
            batch_x = X_p[batch_size * b:batch_size * (b + 1),:]
            batch_y = Y_p[batch_size * b:batch_size * (b + 1),:]
            feed_dict = {x_: batch_x, y_: batch_y}
            
            summary, _ = sess.run([merged, train_op],
                                  feed_dict=feed_dict,
                                  options=run_options,
                                  run_metadata=run_metadata)
            train_writer.add_summary(summary, epoch * total_batches + b)
        if run_metadata:
            train_writer.add_run_metadata(run_metadata, 'epoch %d' % epoch)
        duration = time.time() - start_time
        logger.debug('Epoch %d done for %.2f secs', epoch, duration)

        if epoch % 10 == 0:
            summary, train_cber, train_ber, train_ce = sess.run([merged, cber, ber, loss], {x_: X_train, y_: Y_train})
            test_writer.add_summary(summary, epoch)
            logger.info('TRAIN epoch: %d CE: %.5f column BER: [%s] (mean: %.5f)', 
                        epoch, train_ce, ','.join('%.5f' % c for c in train_cber), train_ber)
        
            val_cber, val_ber, val_ce = sess.run([cber, ber, loss], {x_: X_valid, y_: Y_valid})
            logger.info('VALIDATION epoch: %d CE: %.5f column BER: [%s] (mean: %.5f)', 
                        epoch, val_ce, ','.join('%.5f' % c for c in val_cber), val_ber)
            
        if epoch % 100 == 0 and epoch > min_iterations:
            logger.info('Saving model at step %d .......', epoch)
            save_path = saver.save(sess, args.model_filename, global_step = epoch)
            logger.info('ok')

    logger.info('Done training!')
    train_writer.close()
    test_writer.close()

    logger.info('Reading final test data...')
    X_test, Y_test = utils.read_data('data/1/y_big_test.csv', 'data/1/b_big_test.csv')

    test_predictions = sess.run(logits, {x_: X_test, y_: Y_test})
    test_predictions = (test_predictions >= threshold).astype(float)
    test_cber = quality.column_bit_error_rate(test_predictions, Y_test)
    test_ber = np.mean(test_cber)
    logger.info('TEST column BER: [%s] (mean: %.5f)', ','.join('%.5f' % c for c in test_cber), test_ber)

    save_path = saver.save(sess, args.model_filename, global_step = max_iterations)
    logger.info('Model stored in %s' % save_path)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning-rate', help='Initial learning rate', type=float, default=0.01)
    parser.add_argument('--max-iterations', help='Number of epochs to run trainer', type=int, default=100)
    parser.add_argument('--batch-size', help='Batch size', type=int, default=100)
    parser.add_argument('--n-shared', help='Size of shared hidden layer', type=int, default=32)
    parser.add_argument('--n-hidden', help='Size of separate hidden layer', type=int, default=16)
    parser.add_argument('--model-filename', help='Path to save model', default='models/shared-bit')
    parser.add_argument('--log-dir', help='Path to save tensorboard logs', default='logs/tensorboard/shared-bit')
    args = parser.parse_args()
    main(args)
