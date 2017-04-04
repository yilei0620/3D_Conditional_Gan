
import theano
import theano.tensor as T
from theano.sandbox.cuda.basic_ops import (as_cuda_ndarray_variable,
                                           host_from_gpu,
                                           gpu_contiguous, HostFromGpu,
                                           gpu_alloc_empty)
from theano.sandbox.cuda.dnn import GpuDnnConvDesc, GpuDnnConv, GpuDnnConvGradI, dnn_conv, dnn_pool
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.nnet.abstract_conv import conv3d_grad_wrt_inputs
from theano.tensor.nnet import conv3d

from rng import t_rng

t_rng = RandomStreams()

def l2normalize(x, axis=1, e=1e-8, keepdims=True):
    return x/l2norm(x, axis=axis, e=e, keepdims=keepdims)

def l2norm(x, axis=1, e=1e-8, keepdims=True):
    return T.sqrt(T.sum(T.sqr(x), axis=axis, keepdims=keepdims) + e)

def cosine(x, y):
    d = T.dot(x, y.T)
    d /= l2norm(x).dimshuffle(0, 'x')
    d /= l2norm(y).dimshuffle('x', 0)
    return d

def euclidean(x, y, e=1e-8):
    xx = T.sqr(T.sqrt((x*x).sum(axis=1) + e))
    yy = T.sqr(T.sqrt((y*y).sum(axis=1) + e))
    dist = T.dot(x, y.T)
    dist *= -2
    dist += xx.dimshuffle(0, 'x')
    dist += yy.dimshuffle('x', 0)
    dist = T.sqrt(dist)
    return dist

def dropout(X, p=0.):
    """
    dropout using activation scaling to avoid test time weight rescaling
    """
    if p > 0:
        retain_prob = 1 - p
        X *= t_rng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    return X

def conv_cond_concat(x, y):
    """ 
    concatenate conditioning vector on feature map axis 
    """
    return T.concatenate([x, y*T.ones((x.shape[0], y.shape[1], x.shape[2], x.shape[3], x.shape[4]))], axis=1)

def batchnorm(X ,g = None, b = None, u=None, s=None, a=1., e=1e-7):
    """
    batchnorm with support for not using scale and shift parameters
    as well as inference values (u and s) and partial batchnorm (via a)
    will detect and use convolutional or fully connected version
    """
    if X.ndim == 5:
        if u is not None and s is not None:
            b_u = u.dimshuffle('x', 0, 'x', 'x', 'x')
            b_s = s.dimshuffle('x', 0, 'x', 'x', 'x')
        else:
            b_u = T.mean(X, axis=[0, 2, 3, 4]).dimshuffle('x', 0, 'x', 'x', 'x')
            b_s = T.mean(T.sqr(X - b_u), axis=[0, 2, 3, 4]).dimshuffle('x', 0, 'x', 'x', 'x')
        if a != 1:
            b_u = (1. - a)*0. + a*b_u
            b_s = (1. - a)*1. + a*b_s
        X = (X - b_u) / T.sqrt(b_s + e)
        if g is not None and b is not None:
            X = X*g.dimshuffle('x', 0, 'x', 'x', 'x') + b.dimshuffle('x', 0, 'x', 'x', 'x')
    elif X.ndim == 2:
        if u is None and s is None:
            u = T.mean(X, axis=0)
            s = T.mean(T.sqr(X - u), axis=0)
        if a != 1:
            u = (1. - a)*0. + a*u
            s = (1. - a)*1. + a*s
        X = (X - u) / T.sqrt(s + e)
        if g is not None and b is not None:
            X = X*g + b
    else:
        raise NotImplementedError
    return X


def conv(X, w, input_shape = None, filter_shape = None, subsample=(2, 2, 2), border_mode=(1,1,1), conv_mode='conv',output_shape = None):
    """ 
    sets up dummy convolutional forward pass and uses its grad as deconv
    currently only tested/working with same padding
    input_shape: (batch size, num input feature maps, voxel height, voxel width, voxel depth)
    filter_shape: (output channels, input channels, filter height, filter width, filter depth)
    """
    if conv_mode == 'conv':
        return conv3d(
                input = X,
                filters = w,
                input_shape = input_shape,
                filter_shape = filter_shape,
                border_mode = border_mode,
                subsample = subsample,
                filter_flip = True
            )
    elif conv_mode == 'deconv':
        if output_shape == None:
            input_shape = (None,None,(input_shape[2]-1)*subsample[0] + filter_shape[2] - 2*border_mode[0]
                    ,(input_shape[3]-1)*subsample[1] + filter_shape[3] - 2*border_mode[0]
                    ,(input_shape[4]-1)*subsample[2] + filter_shape[4] - 2*border_mode[0])
        else:
            input_shape = output_shape

        return conv3d_grad_wrt_inputs(
                output_grad = X,
                filters = w,
                input_shape = input_shape,
                filter_shape = filter_shape,
                border_mode = border_mode,
                subsample = subsample,
                )
