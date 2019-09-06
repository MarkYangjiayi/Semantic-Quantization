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
