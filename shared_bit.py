#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import logging
import numpy as np
import tensorflow as tf
import time

from shared_bit_model import SharedBitModel
import utils
import quality
import dataset


logger = logging.getLogger(__name__)


def prepare_writers(sess, args):
    if args.clean_logs:
        if tf.gfile.Exists(args.log_dir):
            tf.gfile.DeleteRecursively(args.log_dir)
        tf.gfile.MakeDirs(args.log_dir)
    train_writer = tf.summary.FileWriter(args.log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(args.log_dir + '/test')
    return train_writer, test_writer


def main(args):
    sess = tf.Session()

    logger.info('Creating model...')
    with tf.name_scope('input'):
        x_ = tf.placeholder(dtype = tf.float32, shape = (None, utils.N_BITS), name='x-input')
        y_ = tf.placeholder(dtype = tf.float32, shape = (None, utils.N_BITS), name='y-input')


    model = SharedBitModel(args, x_, y_)
    logits = model.get_predictions()
    loss = model.get_loss()
    cber, ber = quality.tf_binary_error_rate(logits, y_, threshold=0.0)
    train_op = model.get_training_operation()
    global_step = model.get_global_step()

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
    train_writer, test_writer = prepare_writers(sess, args)

    logger.info('Reading train data...')
    # train_data = dataset.read_dataset('data/1/ML_noise/y_ml_noise_10.csv', 'data/1/ML_noise/z_ml_noise_10.csv', transposed=False)
    train_data = dataset.read_dataset('data/1/y_10_7.csv', 'data/1/b_10_7.csv', transposed=True)
    logger.info('shape x %s, shape y %s', train_data.X.shape, train_data.Y.shape)
    
    logger.info('Reading validation data...')
    valid_data = dataset.read_dataset('data/1/ML_noise/y_val_ml_noise_10.csv', 'data/1/ML_noise/b_val_ml_noise_10.csv', transposed=False)

    logger.info('Initialize model...')
    sess.run(init_op)

    min_iterations = 0
    if args.min_epoch > 0:
        min_iterations = args.min_epoch
        model_filename = '%s-%s' % (args.model_filename, args.min_epoch)
        logger.info('Restore model of %d epoch from %s', args.min_epoch, model_filename)
        saver.restore(sess, model_filename)
    
    max_iterations = args.max_epochs
    global_iteration = 0
    logger.info('Training...')
    for epoch in xrange(min_iterations, max_iterations):
        start_time = time.time()
        
        for batch in train_data.batches_generator(args.batch_size):
            if global_iteration % 5000 == 4999:
                summary, train_cber, train_ber, train_ce, _, global_iteration = sess.run(
                                                        [merged, cber, ber, loss, train_op, global_step], 
                                                        feed_dict={x_: batch.X, y_: batch.Y})
                logger.info('TRAIN step: %d CE: %.5f column BER: [%s] (mean: %.5f)',
                            global_iteration, train_ce, ','.join('%.5f' % c for c in train_cber), train_ber)
                train_writer.add_summary(summary, global_iteration)
            else:
                _, global_iteration = sess.run([train_op, global_step], feed_dict={x_: batch.X, y_: batch.Y})
        duration = time.time() - start_time
        logger.debug('Epoch %d done for %.2f secs (current global iteration: %d)', epoch, duration, global_iteration)

        if epoch % 5 == 4:
            valid_cber, valid_ber, valid_ce = sess.run([cber, ber, loss], 
                                                       feed_dict={x_: valid_data.X, y_: valid_data.Y})
            logger.info('VALID epoch: %d CE: %.5f column BER: [%s] (mean: %.5f)',
                        epoch, valid_ce, ','.join('%.5f' % c for c in valid_cber), valid_ber)
        
        if epoch % 25 == 0 and epoch > min_iterations:
            logger.info('Saving model at step %d .......', epoch)
            save_path = saver.save(sess, args.model_filename, global_step = epoch)
            logger.info('ok')

    logger.info('Done training!')
    train_writer.close()
    test_writer.close()

    valid_data = dataset.read_dataset('data/1/ML_noise/y_val_ml_noise_10.csv', 'data/1/ML_noise/b_val_ml_noise_10.csv', transposed=False)
    valid_cber, valid_ber, valid_ce = sess.run([cber, ber, loss], 
                                                feed_dict={x_: valid_data.X, y_: valid_data.Y})
    logger.info('VALID CE: %.5f column BER: [%s] (mean: %.5f)',
                       valid_ce, ','.join('%.5f' % c for c in valid_cber), valid_ber)
    with open('noised_bits_db_8x128_8x32.txt', 'w') as f:
        for t in xrange(13):
            logger.info('Reading final test data...%d', t)
            X_test, Y_test = utils.read_data('data/1/db/Y_noise_db_%d.csv' % t, 'data/1/db/b_noise_db_%d.csv' % t, transposed=True)

            test_cber, test_ber, test_ce = sess.run([cber, ber, loss], {x_: X_test, y_: Y_test})
            logger.info('TEST CE %d: %.5f column BER: [%s] (mean: %.5f)', t, 
                    test_ce, ','.join('%.5f' % c for c in test_cber), test_ber)
            print >> f, ','.join('%.5f' % c for c in test_cber)
            logger.info('I am fine!')

    save_path = saver.save(sess, args.model_filename, global_step = max_iterations)
    logger.info('Model stored in %s' % save_path)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning-rate', help='Initial learning rate', type=float, default=0.01)
    parser.add_argument('--min-epoch', help='Epoch to load model from', type=int, default=0)
    parser.add_argument('--max-epochs', help='Number of epochs to run trainer', type=int, default=100)
    parser.add_argument('--batch-size', help='Batch size', type=int, default=100)
    parser.add_argument('--n-shared', help='Size of shared hidden layer', type=int, default=32)
    parser.add_argument('--shared-layers', help='Number of shared hidden layers', type=int, default=1)
    parser.add_argument('--n-hidden', help='Size of separate hidden layer', type=int, default=16)
    parser.add_argument('--finger-layers', help='Number of layers in finger', type=int, default=1)
    parser.add_argument('--l2', help='Regularization coef', type=float, default=0.0)
    parser.add_argument('--model-filename', help='Path to save model', default='models/shared-bit')
    parser.add_argument('--log-dir', help='Path to save tensorboard logs', default='logs/tensorboard/shared-bit')
    parser.add_argument('--clean-logs', help='Clean logs dir', action='store_true')
    args = parser.parse_args()
    main(args)
