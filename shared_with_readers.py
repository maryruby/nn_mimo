#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pandas as pd
import tensorflow as tf


N_BITS = 8


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'received': tf.FixedLenFeature([], tf.float32), # or maybe shape=(1,N_BITS) ?
            'sent': tf.FixedLenFeature([], tf.float32),
        })

    return features['received'], features['sent']


def inputs(filename, batch_size, num_epochs=None):
  """Reads input data num_epochs times

  Args:
    filename: filename with data in tfrecords format
    batch_size: Number of examples per returned batch
    num_epochs: Number of times to read the input data, or 0/None to train forever

  Returns a tuple (received, sent), where:
    * received is a float tensor with shape [batch_size, N_RECEIVING_ANTENNAS]
    * sent is an float tensor with shape [batch_size, N_SENDING_ANTENNAS]
  """
  if not num_epochs: num_epochs = None
  
  with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer(
        [filename], num_epochs=num_epochs)

    # Even when reading in multiple threads, share the filename queue
    received, sent = read_and_decode(filename_queue)

    # Shuffle the examples and collect them into batch_size batches.
    # (Internally uses a RandomShuffleQueue.)
    # We run this in two threads to avoid being a bottleneck.
    X, Y = tf.train.shuffle_batch(
        [received, sent], batch_size=batch_size, num_threads=2,
        capacity=1000 + 3 * batch_size,
        # Ensures a minimum amount of shuffling of examples.
        min_after_dequeue=1000)

    return X, Y


def create_model(X, args):
    """Return output layer tensor"""
    hiddens = [tf.contrib.layers.fully_connected(X, args.layer_size, activation_fn=tf.nn.relu)]
    for i in xrange(args.layers - 1):
        hiddens.append(tf.contrib.layers.fully_connected(hiddens[-1], args.layer_size, activation_fn=tf.nn.relu))
    logits = tf.contrib.layers.fully_connected(hiddens[-1], N_BITS, activation_fn=tf.nn.sigmoid)
    return logits


def loss(logits, labels):
  cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits, name='xentropy')
  return tf.reduce_mean(cross_entropy, name='xentropy_mean')


def training(loss, learning_rate)
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


def run_training(args):
    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        X, Y = inputs(args.train, batch_size=args.batch_size, num_epochs=args.num_epochs)

        # Build a Graph that computes predictions from the model
        logits = create_model(X, args)

        # Add to the Graph the loss calculation
        loss = loss(logits, Y)

        # Add to the Graph operations that train the model.
        train_op = training(loss, args.learning_rate)

        # The op for initializing the variables.
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

        # Create a session for running operations in the Graph.
        sess = tf.Session()

        # Initialize the variables (the trained variables and the
        # epoch counter).
        sess.run(init_op)

        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            step = 0
            while not coord.should_stop():
                start_time = time.time()

                # Run one step of the model.  The return values are
                # the activations from the `train_op` (which is
                # discarded) and the `loss` op.  To inspect the values
                # of your ops or variables, you may include them in
                # the list passed to sess.run() and the value tensors
                # will be returned in the tuple from the call.
                _, loss_value = sess.run([train_op, loss])

                duration = time.time() - start_time

                # Print an overview fairly often.
                if step % 100 == 0:
                    print 'Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration)
                step += 1
        except tf.errors.OutOfRangeError:
            print 'Done training for %d epochs, %d steps.' % (args.num_epochs, step)
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

        # Wait for threads to finish.
        coord.join(threads)
        sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', help='Initial learning rate', type=float, default=0.01)
    parser.add_argument('--num_epochs', help='Number of epochs to run trainer', type=int, default=100)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=100)
    parser.add_argument('--train', help='Directory with the training data')
    parser.add_argument('--layer-size', help='Size of hidden layer')
    parser.add_argument('--layers', help='Number of hidden layers')
    args = parser.parse_args()
    run_training(args)
