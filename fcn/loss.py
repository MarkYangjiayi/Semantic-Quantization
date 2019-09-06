"""This module provides the a softmax cross entropy loss for training FCN.

In order to train VGG first build the model and then feed apply vgg_fcn.up
to the loss. The loss function can be used in combination with any optimizer
(e.g. Adam) to finetune the whole model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def loss(logits, labels, num_classes, head=None, ignore_label=0):
    """Calculate the loss from the logits and the labels.

    Args:
      logits: tensor, float - [batch_size, width, height, num_classes].
          Use vgg_fcn.upscore as logits.
      labels: Labels tensor, int32 - [batch_size, width, height, num_classes].
          The ground truth of your data.
      head: numpy array - [num_classes]
          Weighting the loss of each class
          Optional: Prioritize some classes

    Returns:
      loss: Loss tensor of type float.
    """
    with tf.name_scope('loss'):
        # print(logits)
        # logits = tf.reshape(logits, (-1, num_classes))
        # epsilon = tf.constant(value=1e-4)
        # # labels = tf.argmax(labels, dimension=3)
        # # labels = tf.expand_dims(labels, dim=3)
        # labels = tf.contrib.layers.one_hot_encoding(labels,151)
        # labels = tf.to_float(tf.reshape(labels, (-1, num_classes)))
        #
        # softmax = tf.nn.softmax(logits) + epsilon
        #
        # if head is not None:
        #     cross_entropy = -tf.reduce_sum(tf.multiply(labels * tf.log(softmax),
        #                                    head), reduction_indices=[1])
        # else:
        #     cross_entropy = -tf.reduce_sum(
        #         labels * tf.log(softmax), reduction_indices=[1])
        #
        # cross_entropy_mean = tf.reduce_mean(cross_entropy,
        #                                     name='xentropy_mean')
        loss_weight=1.0
        scaled_labels = tf.reshape(labels, shape=[-1])
        not_ignore_mask = tf.to_float(tf.not_equal(scaled_labels,
                                                   ignore_label)) * loss_weight
        one_hot_labels = tf.contrib.layers.one_hot_encoding(
            scaled_labels, num_classes, on_value=1.0, off_value=0.0)
        cross_entropy_mean = tf.losses.softmax_cross_entropy(
            one_hot_labels,
            tf.reshape(logits, shape=[-1, num_classes]),
            weights=not_ignore_mask)

        tf.add_to_collection('losses', cross_entropy_mean)

        loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
    return loss
