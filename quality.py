#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import utils


def bit_error_rate(predicted, real):
    """Returns rate of correctly predicted bits (accuracy) aka BER"""
    total_elems = (real.shape[0] * real.shape[1])
    return np.sum(np.not_equal(predicted, real)) / float(total_elems)


def column_bit_error_rate(predicted, real):
    """Returns numpy array with BER for each column of data"""
    return np.sum(np.not_equal(predicted, real), 0) / float(real.shape[0])


def symbol_error_rate(predicted, real):
    """Returns rate of correctly predicted rows"""
    total_elems = (real.shape[0] * real.shape[1])
    return np.sum(np.any(np.not_equal(predicted, real), 1)) / float(total_elems)


def tf_binary_error_rate(logits, labels, threshold=0.0):
    t = tf.constant(threshold)
    with tf.name_scope('ber'):
        with tf.name_scope('errors'):
            errors = tf.not_equal(tf.cast(tf.greater_equal(logits, t), tf.float32), labels)
        with tf.name_scope('column_ber'):
            column_ber = tf.reduce_mean(tf.cast(errors, tf.float32), axis=0)
            for column in xrange(utils.N_BITS):
                tf.summary.scalar('ber_%d' % column, column_ber[column])
    ber = tf.reduce_mean(column_ber)
    tf.summary.scalar('ber', ber)
    return column_ber, ber
