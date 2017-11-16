import numpy as np
import tensorflow as tf
from model import BaseModel
import utils


def to_binary_tensor(x):
    if x == 0: 
        return tf.constant([0] * utils.N_BITS, dtype=tf.int32)
    bit = []
    while x:
        bit.append(x % 2)
        x >>= 1
    if len(bit) < utils.N_BITS:
        bit = bit + [0] * (utils.N_BITS - len(bit))
    return tf.constant(bit[::-1], dtype=tf.int32)


binary_tensors =  tf.stack([to_binary_tensor(x) for x in xrange(2 ** utils.N_BITS)], axis=0)


class MaxLikelihoodModel(BaseModel):
    def __init__(self, args, x_input, y_input):
        self.args = args
        self.x_input = x_input
        y_int = tf.cast(y_input, dtype=tf.int32)
        digits = tf.reshape(2 ** tf.range(utils.N_BITS - 1, -1, -1, dtype=tf.int32), (utils.N_BITS, 1))
        self.y_input = tf.one_hot(tf.matmul(y_int, digits), 2**utils.N_BITS)

        logits = create_model(self.x_input, self.args)
        self.predictions = tf.argmax(logits, 1)
        self.loss = create_loss(logits, self.y_input)
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
    with tf.name_scope('max_likelihood'):
        hiddens = [x]

        with tf.name_scope('hidden'):
            for hidden_layer in xrange(args.hidden_layers):
                with tf.name_scope('hidden-%d' % hidden_layer):
                    hiddens.append(tf.contrib.layers.fully_connected(hiddens[-1], args.n_hidden, 
                                                                     activation_fn=tf.nn.relu))

        with tf.name_scope('softmax'):
            output = tf.contrib.layers.fully_connected(hiddens[-1], 2**utils.N_BITS,
                                                       activation_fn=tf.identity)
            tf.summary.histogram('activations', output)
        return output


# logits = predicted tensor, labels = ideal tensor;
def create_loss(logits, labels):
    with tf.name_scope('cross_entropy'):
        ce = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        tf.summary.scalar('cross_entropy', ce)
    return ce


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
