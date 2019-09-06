#coding=utf-8
import tensorflow as tf
#bitsQ is not used at the moment
bitsW = 8
bitsA = 8
bitsG = 8
bitsE = 8
bitsU = 24

# bitsW = 32
# bitsA = 32
# bitsG = 32
# bitsE = 32
# bitsU = 32

bitsR = 32

def Shift(x):
    return 2 ** tf.round(tf.log(x) / tf.log(2.0))

def S(bits):
    return 2.0 ** (bits - 1)

#clip function
def clip(x,bits):
    if bits>=32:
        delta=0.0
    else:
        x=tf.cast(x,tf.float32)
        bits=tf.cast(bits,tf.float32)
        delta=1./tf.pow(2.0,bits-1)
    MAX=+1-delta
    MIN=-1+delta
    x=tf.clip_by_value(x,MIN,MAX,name='saturate')
    return x

@tf.custom_gradient
def Q(x, bits):
    def grad(dy):
        return dy
    x=tf.cast(x,tf.float32)
    bits=tf.cast(bits,tf.float32)
    n=tf.pow(2.0,bits-1)#有函数了还要单算？TODO
    y=tf.round(x*n)/n
    return y,grad

@tf.custom_gradient
def Qinit(x, bits):
    def grad(dy):
        return dy
    x=tf.cast(x,tf.float32)
    bits=tf.cast(bits,tf.float32)
    n=tf.pow(2.0,bits-1)#有函数了还要单算？TODO
    y=tf.round(x*n)/n
    return y,grad


@tf.custom_gradient
def QW(x):
    def grad(dy):
        return dy
    if bitsW >=32:
        return x,grad
    else:
        result = clip(Q(x,bitsW),bitsW)
        tf.add_to_collection("quantized_weights",result)
        return result,grad

#quantize activation
@tf.custom_gradient
def QA(x):
    def grad(dy):
        return dy
    if bitsA>=32:
        return x,grad
    else:
        return_x = Q(x,bitsA)
        # return_x = clip(return_x,bitsA)
        return return_x,grad

# @tf.custom_gradient
# def QG(x):
#     tf.summary.histogram(x.op.name + "/gradient",x)
#     def grad(dy):
#       return dy
#     if bitsG>=32:
#       tf.summary.histogram(x.op.name + "/gradient_Q",x)
#       return x, grad
#     else:
#       return Q(x,bitsG),grad

@tf.custom_gradient
def QG(x, LR=1.):
    tf.summary.histogram(x.op.name + "/gradient",x)
    def grad(dy):
        return dy
    if bitsG>=32:
        return x,grad
    else:

        g_scale = 128
        xmax = tf.reduce_max(tf.abs(x))
        xavg = tf.reduce_mean(tf.abs(x))
        tf.add_to_collection("G_scale", xmax)#value extraction
        x = x / Shift(xmax)

        norm = Q(g_scale * x, bitsR)
        norm_sign = tf.sign(norm)
        norm_abs = tf.abs(norm)
        norm_int = tf.floor(norm_abs)
        norm_float = norm_abs - norm_int
        rand_float = tf.random_uniform(x.get_shape(), 0, 1)
        norm = norm_sign * ( norm_int + 0.5 * (tf.sign(norm_float - rand_float) + 1))
        result = norm / (g_scale*S(bitsG))
        result = Q(LR*result,bitsU)#added on 7.28

        # result = (norm * Shift(xmax))/ (g_scale)#encoder大概被缩小了10倍
        # result = tf.clip_by_value(result,-10,10)
        # result = Q(LR*result,bitsU)

        # result = (norm * Shift(xmax))/ (g_scale)#encoder大概被缩小了10倍
        # result = tf.clip_by_value(Q(LR*result,bitsU),-1,1)
        # tf.summary.histogram(x.op.name + "/gradient_Q",result)
        return result,grad

#quantize error 1
@tf.custom_gradient
def QE(x):
    def grad(dy):
        tf.summary.histogram("error",dy)
        if bitsE>=32:
            tf.summary.histogram("error_Q",dy)
            return dy
        else:
            dymax = tf.reduce_max(tf.abs(dy))
            dymax_shift = Shift(dymax)
            # return  dymax_shift*tf.clip_by_value(Q(dy /dymax_shift, bitsE),-1,1)
            return  dymax_shift*clip(Q(dy /dymax_shift, bitsE),bitsE)

            # dymax = tf.reduce_max(tf.abs(dy))
            # dymax = Q(dymax,32)
            # return  dymax*tf.clip_by_value(Q(dy /dymax, bitsE),-1,1)
            # return Q(clip(x /dymax_shift, bitsE), bitsE)
    return x,grad



#record: naive QG with 8 bits and scale 2 can train normally
