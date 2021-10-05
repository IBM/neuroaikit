"""Activations provided by NeuroAI Toolkit.
"""

import tensorflow as tf


def step_function(x, pseudoderivative_of=tf.nn.tanh):
    """Step function activation that in backward pass acts as if it was a function given in the
    `pseudoderivative_of` parameter.

    :param x:
    :param pseudoderivative_of: function to take derivative of. Defaults to: tf.nn.tanh
    :return:
    """
    # forward pass: step function (entire expression is evaluated)
    # backward pass: tanh derivative - triangle (-2,2) (only the first part is considered)
    return pseudoderivative_of(x) + tf.stop_gradient(-pseudoderivative_of(x) + tf.nn.relu(tf.sign(x)))


def leaky_rel(x, alpha=0.1):
    """Leaky_relu activation with alpha changed to 0.1 by default.
    """
    return tf.nn.leaky_relu(x, alpha=alpha)
