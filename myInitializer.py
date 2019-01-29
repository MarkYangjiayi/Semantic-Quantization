"""Weight initializers for use with layers."""
import math
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import random_ops
import Quantize
import Option


def variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False,
                                 seed=None, dtype=dtypes.float32):
  """Returns an initializer that generates tensors without scaling variance.

  When initializing a deep network, it is in principle advantageous to keep
  the scale of the input variance constant, so it does not explode or diminish
  by reaching the final layer. This initializer use the following formula:

  ```python
    if mode='FAN_IN': # Count only number of input connections.
      n = fan_in
    elif mode='FAN_OUT': # Count only number of output connections.
      n = fan_out
    elif mode='FAN_AVG': # Average number of inputs and output connections.
      n = (fan_in + fan_out)/2.0

      truncated_normal(shape, 0.0, stddev=sqrt(factor / n))
  ```

  * To get [Delving Deep into Rectifiers](
     http://arxiv.org/pdf/1502.01852v1.pdf), use (Default):<br/>
    `factor=2.0 mode='FAN_IN' uniform=False`
  * To get [Convolutional Architecture for Fast Feature Embedding](
     http://arxiv.org/abs/1408.5093), use:<br/>
    `factor=1.0 mode='FAN_IN' uniform=True`
  * To get [Understanding the difficulty of training deep feedforward neural
    networks](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf),
    use:<br/>
    `factor=1.0 mode='FAN_AVG' uniform=True.`
  * To get `xavier_initializer` use either:<br/>
    `factor=1.0 mode='FAN_AVG' uniform=True`, or<br/>
    `factor=1.0 mode='FAN_AVG' uniform=False`.

  Args:
    factor: Float.  A multiplicative factor.
    mode: String.  'FAN_IN', 'FAN_OUT', 'FAN_AVG'.
    uniform: Whether to use uniform or normal distributed random initialization.
    seed: A Python integer. Used to create random seeds. See
      [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
      for behavior.
    dtype: The data type. Only floating point types are supported.

  Returns:
    An initializer that generates tensors with unit variance.

  Raises:
    ValueError: if `dtype` is not a floating point type.
    TypeError: if `mode` is not in ['FAN_IN', 'FAN_OUT', 'FAN_AVG'].
  """
  if not dtype.is_floating:
    raise TypeError('Cannot create initializer for non-floating point type.')
  if mode not in ['FAN_IN', 'FAN_OUT', 'FAN_AVG']:
    raise TypeError('Unknow mode %s [FAN_IN, FAN_OUT, FAN_AVG]', mode)

  # pylint: disable=unused-argument
  def _initializer(shape, dtype=dtype, partition_info=None):
    """Initializer function."""
    if not dtype.is_floating:
      raise TypeError('Cannot create initializer for non-floating point type.')
    # Estimating fan_in and fan_out is not possible to do perfectly, but we try.
    # This is the right thing for matrix multiply and convolutions.


    if shape:
      fan_in = float(shape[-2]) if len(shape) > 1 else float(shape[-1])
      fan_out = float(shape[-1])
    else:
      fan_in = 1.0
      fan_out = 1.0
    for dim in shape[:-2]:
      fan_in *= float(dim)
      fan_out *= float(dim)
    if mode == 'FAN_IN':
      # Count only number of input connections.
      n = fan_in
    elif mode == 'FAN_OUT':
      # Count only number of output connections.
      n = fan_out
    elif mode == 'FAN_AVG':
      # Average number of inputs and output connections.
      n = (fan_in + fan_out) / 2.0


    def scaleLimit(limit):
      bitsW = Quantize.bitsW
      scale = 1.0
      if bitsW < 32:
        beta = 1.5
        Wm = beta / Quantize.S(bitsW)
        scale = 2 ** round(math.log(Wm / limit, 2.0))
        scale = scale if scale > 1 else 1.0
        limit = Wm if Wm > limit else limit
      Option.W_scale.append(scale)
      return limit

    if uniform:
      # To get stddev = math.sqrt(factor / n) need to adjust for uniform.
      limit = scaleLimit(math.sqrt(3.0 * factor / n))
      return random_ops.random_uniform(shape, -limit, limit,dtype, seed=seed)

    else:
      # To get stddev = math.sqrt(factor / n) need to adjust for truncated.
      trunc_stddev = scaleLimit(math.sqrt(1.3 * factor / n))
      return random_ops.truncated_normal(shape, 0.0, trunc_stddev, dtype,
                                         seed=seed)
  # pylint: enable=unused-argument

  return _initializer
