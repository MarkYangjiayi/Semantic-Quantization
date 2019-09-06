import tensorflow as tf


_MEAN_RGB = [123.15, 115.90, 103.06]

def crop_images(image,label,crop_size,ignore_label):
    processed_image = tf.cast(image, tf.float32)
    label = tf.cast(label, tf.float32)
    image_shape = tf.shape(image)
    crop_height = crop_size[0]
    crop_width = crop_size[1]
    image_height = image_shape[0]
    image_width = image_shape[1]

    target_height = image_height + tf.maximum(crop_height - image_height, 0)
    target_width = image_width + tf.maximum(crop_width - image_width, 0)

    mean_pixel = tf.reshape(
        _MEAN_RGB, [1, 1, 3])
    mean_pixel =tf.cast(mean_pixel,dtype=tf.uint8)
    image = pad_to_bounding_box(
        image, 0, 0, target_height, target_width, mean_pixel)
    label = pad_to_bounding_box(
        label, 0, 0, target_height, target_width, ignore_label)

    # image = tf.image.random_crop(image,[crop_height,crop_width,3])
    # label = tf.image.random_crop(label,[crop_height,crop_width,1])
    image = crop(image,0,0,crop_height,crop_width)
    label = crop(label,0,0,crop_height,crop_width)
    return image, label

def pad_to_bounding_box(image, offset_height, offset_width, target_height,
                        target_width, pad_value):
  """Pads the given image with the given pad_value.

  Works like tf.image.pad_to_bounding_box, except it can pad the image
  with any given arbitrary pad value and also handle images whose sizes are not
  known during graph construction.

  Args:
    image: 3-D tensor with shape [height, width, channels]
    offset_height: Number of rows of zeros to add on top.
    offset_width: Number of columns of zeros to add on the left.
    target_height: Height of output image.
    target_width: Width of output image.
    pad_value: Value to pad the image tensor with.

  Returns:
    3-D tensor of shape [target_height, target_width, channels].

  Raises:
    ValueError: If the shape of image is incompatible with the offset_* or
    target_* arguments.
  """
  image_rank = tf.rank(image)
  image_rank_assert = tf.Assert(
      tf.equal(image_rank, 3),
      ['Wrong image tensor rank [Expected] [Actual]',
       3, image_rank])
  with tf.control_dependencies([image_rank_assert]):
    image -= pad_value
  image_shape = tf.shape(image)
  height, width = image_shape[0], image_shape[1]
  target_width_assert = tf.Assert(
      tf.greater_equal(
          target_width, width),
      ['target_width must be >= width'])
  target_height_assert = tf.Assert(
      tf.greater_equal(target_height, height),
      ['target_height must be >= height'])
  with tf.control_dependencies([target_width_assert]):
    after_padding_width = target_width - offset_width - width
  with tf.control_dependencies([target_height_assert]):
    after_padding_height = target_height - offset_height - height
  offset_assert = tf.Assert(
      tf.logical_and(
          tf.greater_equal(after_padding_width, 0),
          tf.greater_equal(after_padding_height, 0)),
      ['target size not possible with the given target offsets'])

  height_params = tf.stack([offset_height, after_padding_height])
  width_params = tf.stack([offset_width, after_padding_width])
  channel_params = tf.stack([0, 0])
  with tf.control_dependencies([offset_assert]):
    paddings = tf.stack([height_params, width_params, channel_params])
  padded = tf.pad(image, paddings)
  return padded + pad_value

def crop(image, offset_height, offset_width, crop_height, crop_width):
  """Crops the given image using the provided offsets and sizes.

  Note that the method doesn't assume we know the input image size but it does
  assume we know the input image rank.

  Args:
    image: an image of shape [height, width, channels].
    offset_height: a scalar tensor indicating the height offset.
    offset_width: a scalar tensor indicating the width offset.
    crop_height: the height of the cropped image.
    crop_width: the width of the cropped image.

  Returns:
    The cropped (and resized) image.

  Raises:
    ValueError: if `image` doesn't have rank of 3.
    InvalidArgumentError: if the rank is not 3 or if the image dimensions are
      less than the crop size.
  """
  original_shape = tf.shape(image)

  if len(image.get_shape().as_list()) != 3:
    raise ValueError('input must have rank of 3')
  original_channels = image.get_shape().as_list()[2]

  rank_assertion = tf.Assert(
      tf.equal(tf.rank(image), 3),
      ['Rank of image must be equal to 3.'])
  with tf.control_dependencies([rank_assertion]):
    cropped_shape = tf.stack([crop_height, crop_width, original_shape[2]])

  size_assertion = tf.Assert(
      tf.logical_and(
          tf.greater_equal(original_shape[0], crop_height),
          tf.greater_equal(original_shape[1], crop_width)),
      ['Crop size greater than the image size.'])

  offsets = tf.to_int32(tf.stack([offset_height, offset_width, 0]))

  # Use tf.slice instead of crop_to_bounding box as it accepts tensors to
  # define the crop size.
  with tf.control_dependencies([size_assertion]):
    image = tf.slice(image, offsets, cropped_shape)
  image = tf.reshape(image, cropped_shape)
  image.set_shape([crop_height, crop_width, original_channels])
  return image
