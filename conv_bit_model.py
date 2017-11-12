import numpy as np
import tensorflow as tf
from model import BaseModel
import utils


class ConvBitModel(BaseModel):
    def __init__(self, args, x_input, y_input, train_input):
        self.args = args
        self.x_input = x_input
        self.y_input = y_input
        self.predictions = create_model(self.x_input, self.args, train_input)
        self.loss = create_loss(self.predictions, self.y_input)
        self.train_op, self.global_step = training(self.loss, self.args)

    def get_predictions(self):
        return self.predictions

    def get_loss(self):
        return self.loss

    def get_training_operation(self):
        return self.train_op

    def get_global_step(self):
        return self.global_step


def create_model(x, args, train_input):
    with tf.name_scope('conv_bit'):
        with tf.variable_scope('conv1') as scope:
            conv1 = tf.layers.conv1d(tf.expand_dims(x, -1),
                                    args.conv1_kernels,
                                    kernel_size=args.conv1_size,
                                    strides=args.conv1_size,
                                    padding='VALID',
                                    activation=tf.nn.relu)
            tf.summary.histogram('activations', conv1)
            tf.summary.scalar('sparsity', tf.nn.zero_fraction(x))

        # conv1 shape will be (?, 4, 8), but we need to flatten last two dimensions
        shared_hiddens = [tf.reshape(conv1, [tf.shape(conv1)[0], 8 / args.conv1_size * args.conv1_kernels])]
        print shared_hiddens[-1].get_shape()

        with tf.name_scope('shared'):
            for sh in xrange(args.shared_layers):
                with tf.name_scope('%d' % sh):
                    fully1 = tf.contrib.layers.fully_connected(shared_hiddens[-1], args.n_shared, activation_fn=tf.nn.relu)
                    do1 = tf.layers.dropout(inputs=fully1,
                                               rate=args.shared_dropout,
                                               training=train_input)
                    shared_hiddens.append(do1)


        with tf.name_scope('fingers'):
            outputs = []
            for i in range(utils.N_BITS):
                with tf.name_scope('hidden_%d' % i):
                    inners = [shared_hiddens[-1]]
                    for shf in xrange(args.finger_layers):
                        with tf.name_scope('%d' % shf):
                            fully2 = tf.contrib.layers.fully_connected(inners[-1], args.n_hidden, activation_fn=tf.nn.relu)
                            do2 = tf.layers.dropout(inputs=fully2,
                                                      rate=args.finger_dropout,
                                                      training=train_input)
                            inners.append(do2)

                print len(inners)
                with tf.name_scope('output_%d' % i):
                    fully3 = tf.contrib.layers.fully_connected(inners[-1], 1, activation_fn=tf.identity)
                    outputs.append(fully3)
                    tf.summary.histogram('activations', outputs[-1])
        with tf.name_scope('activations'):
            # after stack we will get shape (?, N_BITS, 1), then we squeeze it to (?, N_BITS)
            predictions = tf.squeeze(tf.stack(outputs, axis=1))
            tf.summary.histogram('activations', predictions)
        return predictions


# logits = predicted tensor, labels = ideal tensor;
def create_loss(logits, labels):
    with tf.name_scope('cross_entropy'):
        ce = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits), axis=0)
        for column in xrange(utils.N_BITS):
            tf.summary.scalar('ce_%d' % column, ce[column])
        cross_entropy = tf.reduce_sum(ce)
    tf.summary.scalar('cross_entropy', cross_entropy)
    return cross_entropy


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
        return train_op, global_step