#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import logging
import numpy as np
import tensorflow as tf
import time

from max_likelihood_model import MaxLikelihoodModel, binary_tensors
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


    model = MaxLikelihoodModel(args, x_, y_)
    predictions = model.get_predictions()
    loss = model.get_loss()
    train_op = model.get_training_operation()
    global_step = model.get_global_step()

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    saver = tf.train.Saver()
    logger.info('Initialize model...')
    sess.run(init_op)

    b_tensors = sess.run(binary_tensors)

    if args.min_epoch > 0:
        model_filename = '%s-%s' % (args.model_filename, args.min_epoch)
        logger.info('Restore model of %d epoch from %s', args.min_epoch, model_filename)
        saver.restore(sess, model_filename)

    if args.min_epoch + 1 >= args.max_epoch:
        logger.info('Do not train, just apply model...')
    else:
        merged = tf.summary.merge_all()
        train_writer, test_writer = prepare_writers(sess, args)

        logger.info('Reading train data...')
        train_data = dataset.read_dataset('data/1/ML_noise/y_ml_noise_10.csv', 'data/1/ML_noise/b_ml_noise_10.csv', transposed=False)
        # train_data = dataset.read_dataset('data/1/y_10_7.csv', 'data/1/b_10_7.csv', transposed=False)
        logger.info('shape x %s, shape y %s', train_data.X.shape, train_data.Y.shape)
        
        logger.info('Reading validation data...')
        valid_data = dataset.read_dataset('data/1/ML_noise/y_val_ml_noise_10.csv', 'data/1/ML_noise/b_val_ml_noise_10.csv', transposed=False)
        
        global_iteration = 0
        logger.info('Training...')
        prev_train_ce = sess.run(loss, feed_dict={x_: train_data.X, y_: train_data.Y})
        try:
            for epoch in xrange(args.min_epoch + 1, args.max_epoch):
                start_time = time.time()
                
                for batch in train_data.batches_generator(args.batch_size):
                    if args.verbose and global_iteration % 5000 == 4999:
                        summary, train_ce, _, global_iteration = sess.run([merged, loss, train_op, global_step], 
                                                                feed_dict={x_: batch.X, y_: batch.Y})
                        logger.info('TRAIN step: %d CE: %.5f', global_iteration, train_ce)
                        train_writer.add_summary(summary, global_iteration)
                    else:
                        _, global_iteration = sess.run([train_op, global_step], feed_dict={x_: batch.X, y_: batch.Y})
                duration = time.time() - start_time
                logger.debug('Epoch %d done for %.2f secs (current global iteration: %d)', epoch, duration, global_iteration)

                if epoch >= 0:
                    net_outputs, valid_ce = sess.run([predictions, loss], 
                                                     feed_dict={x_: valid_data.X, y_: valid_data.Y})
                    predicted = b_tensors[net_outputs]
                    valid_cber = quality.column_bit_error_rate(predicted, valid_data.Y)
                    valid_ser = quality.symbol_error_rate(predicted, valid_data.Y)
                    logger.info('VALID epoch: %d CE: %.5f SER: %.7f column BER: [%s] (mean: %.5f)',
                                epoch, valid_ser, valid_ce, ','.join('%.5f' % c for c in valid_cber), np.mean(valid_cber))
                
                if epoch % 10 == 0 and epoch > args.min_epoch:
                    logger.info('Saving model at step %d .......', epoch)
                    save_path = saver.save(sess, args.model_filename, global_step = epoch)
                    logger.info('ok')
        except KeyboardInterrupt:
            logger.warn('Interrupted!')
        logger.info('Done training!')
        train_writer.close()
        test_writer.close()

        save_path = saver.save(sess, args.model_filename, global_step = epoch)
        logger.info('Model stored in %s', save_path)
    
    logger.info('Apply model on noised test data')
    test_filename = args.test_filename
    if not test_filename:
        test_filename = 'noised_bits_db_%dx%d.txt' % (args.hidden_layers, args.n_hidden)
    logger.info('Write results in %s', test_filename)
    with open(test_filename, 'w') as f:
        for t in xrange(13):
            logger.info('Reading final test data...%d', t)
            X_test, Y_test = utils.read_data('data/1/db/Y_noise_db_%d.csv' % t, 'data/1/db/b_noise_db_%d.csv' % t, transposed=False)

            net_outputs, test_ce = sess.run([predictions, loss], 
                                            feed_dict={x_: X_test, y_: Y_test})
            predicted = b_tensors[net_outputs]
            test_cber = quality.column_bit_error_rate(predicted, Y_test)
            test_ser = quality.symbol_error_rate(predicted, Y_test)
            logger.info('TEST CE %d: %.5f SER: %.7f column BER: [%s] (mean: %.5f)', t, 
                    test_ce, test_ser, ','.join('%.5f' % c for c in test_cber), np.mean(test_cber))
            print >> f, ','.join('%.5f' % c for c in test_cber)
    logger.info('I am done!')


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning-rate', help='Initial learning rate', type=float, default=0.01)
    parser.add_argument('--min-epoch', help='Epoch to load model from', type=int, default=0)
    parser.add_argument('--max-epoch', help='Number of epochs to run trainer', type=int, default=100)
    parser.add_argument('--batch-size', help='Batch size', type=int, default=100)
    parser.add_argument('--n-shared', help='Size of shared hidden layer', type=int, default=32)
    parser.add_argument('--hidden-layers', help='Number of hidden layers', type=int, default=1)
    parser.add_argument('--n-hidden', help='Size of each hidden layer', type=int, default=2**utils.N_BITS)
    parser.add_argument('--model-filename', help='Path to save model', default='models/max-likelihood')
    parser.add_argument('--log-dir', help='Path to save tensorboard logs', default='logs/tensorboard/max-likelihood')
    parser.add_argument('--clean-logs', help='Clean logs dir', action='store_true')
    parser.add_argument('-v', '--verbose', help='Verbose (tensorboard and train outputs)', action='store_true')
    parser.add_argument('--test-filename', help='Path to save noised test result')
    args = parser.parse_args()
    main(args)
