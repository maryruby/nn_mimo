import numpy as np
import tensorflow as tf
from model import BaseModel
import utils


class SharedBitModel(BaseModel):
    def __init__(self, args, x_input, y_input):
        self.args = args
        self.x_input = x_input
        self.y_input = y_input
        self.predictions = create_model(self.x_input, self.args)
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
                                                                 activation_fn=tf.identity,
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