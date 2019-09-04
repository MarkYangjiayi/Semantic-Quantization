# -*- coding: utf-8 -*-
import tensorflow as tf

bitsW=8
bitsA=8
bitsG=8

bitsE2=16#进入BN之前的QE
bitsBN_G=8
bitsBN_B=8
bitsBN_mean=16
bitsBN_var=16
bitsBN_x=16
bits_gBN=8

bitsE1=8#BN输出的,跟着relu的QEBN

bitsU=24
bitsLR=32
# ------------------------------
# bitsW=32
# bitsA=32
# bitsG=32
#
# bitsE2=32#进入BN之前的QE
# bitsBN_G=32
# bitsBN_B=32
# bitsBN_mean=32
# bitsBN_var=32
# bitsBN_x=32
# bits_gBN=32
#
# bitsE1=32#BN输出的,跟着relu的QEBN
#
#bitsU=32
# bitsLR=32

def Shift(x):
    xshift = tf.reduce_max(tf.abs(x))
    return xshift

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
    n=tf.pow(2.0,bits-1)
    y=tf.round(x*n)/n
    return y,grad


@tf.custom_gradient
def QW(x):
    tf.add_to_collection("weights_full", x)
    def grad(dy):
        return dy
    if bitsW >=32:
        return x,grad
    else:
        x = clip(Q(x,bitsW),bitsW)
        tf.add_to_collection("weights_Q", x)
        return x,grad

#quantize activation
@tf.custom_gradient
def QA(x):
    def grad(dy):
        return dy
    if bitsA>=32:
        return x,grad
    else:
        xmax_shift = Shift(x)
        result = xmax_shift*clip(Q(x /xmax_shift, bitsA),bitsA)
        return result ,grad

@tf.custom_gradient
def QG(x,LR=1.):
    def grad(dy):
        return dy
    if bitsG>=32:
        return LR*x,grad #g_scale is learning rate
    else:
        bitsR = 32
        xmax = Shift(x)
        tf.add_to_collection("G_scale", xmax)#value extraction
        x = x / xmax

        g_scale = 128
        norm = Q(g_scale * x, bitsR)

        norm_sign = tf.sign(norm)
        norm_abs = tf.abs(norm)
        norm_int = tf.floor(norm_abs)
        norm_float = norm_abs - norm_int
        rand_float = tf.random_uniform(x.get_shape(), 0, 1)
        norm = norm_sign * (norm_int + 0.5 * (tf.sign(norm_float - rand_float) + 1))
        norm = tf.clip_by_value(norm,-g_scale+1,g_scale-1)
        result = (norm * xmax * 0.1) / (g_scale)
        update = Q(LR*result,bitsU)
        return update, grad

@tf.custom_gradient
def QE(x):
    def grad(dy):
        if bitsE2>=32:
            return dy
        else:
            dymax_shift = Shift(dy)
            result = dymax_shift*clip(Q(dy /dymax_shift, bitsE2),bitsE2)#TODO:在wageubn中这块的上下限变成了政正负1
            tf.summary.histogram(result.op.name + "/error",result)
            return result
    return x,grad

@tf.custom_gradient
def QBNG(x):
    def grad(dy):
        return dy
    if bitsBN_G>=32:
        return x,grad
    else:
        return 2*Q(x/2,bitsBN_G),grad

@tf.custom_gradient
def QBNB(x):
    def grad(dy):
        return dy
    if bitsBN_B>=32:
        return x,grad
    else:
        return 2*Q(x/2,bitsBN_B),grad

@tf.custom_gradient
def QBNM(x):
    def grad(dy):
        return dy
    if bitsBN_mean>=32:
        return x,grad
    else:
        return 2*Q(x/2,bitsBN_mean),grad

@tf.custom_gradient
def QBNV(x):
    def grad(dy):
        return dy
    if bitsBN_var>=32:
        return x,grad
    else:
        return 2*Q(x/2,bitsBN_var),grad

@tf.custom_gradient
def QBNX(x):
    def grad(dy):
        return dy
    if bitsBN_x>=32:
        return x,grad
    else:
        # result = Q(x,bitsBN_x)
        xmax_shift = Shift(x)
        result = xmax_shift*clip(Q(x /xmax_shift, bitsA),bitsA)
        return result,grad

#quantize error activation
@tf.custom_gradient
def QEBN(x):
    def grad(dy):
        if bitsE1>=32:
            return dy
        else:
            dymax_shift = Shift(dy)
            return  dymax_shift*tf.clip_by_value(Q(dy /dymax_shift, bitsE1),-1,1)
    return x,grad

@tf.custom_gradient
def QLR(x):
    def grad(dy):
        return dy
    if bitsLR >=32:
        return x,grad
    else:
        return clip(Q(x,bitsLR),bitsLR),grad

def QBits(x,bits=32):
    if  bits >=32:
       return x
    else:
       return Q(x,bits)
