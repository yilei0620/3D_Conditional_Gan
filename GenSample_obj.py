import sys
sys.path.append('..')

import os
import json
from time import time
import numpy as np
from sklearn.externals import joblib
import scipy
from scipy import io

# from matplotlib import pyplot as plt
# from sklearn.externals import joblib

import theano
import theano.tensor as T

from lib import activations
from lib import updates
from lib import inits
from lib.rng import py_rng, np_rng
from lib.ops import batchnorm, conv_cond_concat, conv, dropout
from lib.theano_utils import floatX, sharedX
from lib.data_utils import OneHot, shuffle, iter_data
from lib.metrics import nnc_score, nnd_score

from load import load_shapenet_train, load_shapenet_test




relu = activations.Rectify()
sigmoid = activations.Sigmoid()
lrelu = activations.LeakyRectify()
bce = T.nnet.binary_crossentropy


parameters = {'objectNumber': 2, 'Nz' : 200, 'Channel' :(1,64,128,256,512),  'kernal':(4,4,4,4), 'batchsize': 50, 'Convlayersize':(64,32,16,8,4), 'Genlrt' : 0.001, 'Discrimlrt' : 0.00001 , 'beta' : 0.5, 'l2':2.5e-5, 'Genk' : 2 , 'niter':50, 'niter_decay' : 150}

for p in parameters:
    tmp = p + " = parameters[p]"
    exec(tmp)

# print conditional,type(batchsize),Channel[-1],kernal


gifn = inits.Normal(scale=0.02)
difn = inits.Normal(scale=0.02)


## filter_shape: (output channels, input channels, filter height, filter width, filter depth)

## load the parameters

# gen_params = [gw1, gw2, gw3, gw4, gw5, gwx]
# discrim_params = [dw1, dw2, dw3, dw4, dw5, dwy]

temp = joblib.load('models%d/50_gen_params.jl'%objectNumber)
gw1 = sharedX(temp[0])
gg1 = sharedX(temp[1])
gb1 = sharedX(temp[2])
gw2 = sharedX(temp[3])
gg2 = sharedX(temp[4])
gb2 = sharedX(temp[5])
gw3 = sharedX(temp[6])
gg3 = sharedX(temp[7])
gb3 = sharedX(temp[8])
gw4 = sharedX(temp[9])
gg4 = sharedX(temp[10])
gb4 = sharedX(temp[11])
gwx = sharedX(temp[12])

gen_params = [gw1, gg1, gb1, gw2, gg2, gb2, gw3, gg3, gb3, gw4 ,gg4, gb4, gwx]

##

def gen(Z, w1, g1, b1, w2, g2, b2, w3, g3, b3, w4, g4, b4, wx):
    Gl1 = relu(batchnorm(T.dot(Z, w1), g=g1, b=b1))
    Gl1 = Gl1.reshape((Gl1.shape[0],Channel[-1],Convlayersize[-1],Convlayersize[-1],Convlayersize[-1]))

    input_shape =  (None , None,Convlayersize[-1],Convlayersize[-1],Convlayersize[-1])
    filter_shape = (Channel[-1] , Channel[-2], kernal[-1], kernal[-1], kernal[-1])

    Gl2 = relu(batchnorm(conv(Gl1,w2,filter_shape = filter_shape, input_shape = input_shape, conv_mode = 'deconv'),g = g2, b = b2))

    input_shape =  (None , None,Convlayersize[-2],Convlayersize[-2],Convlayersize[-2])
    filter_shape = (Channel[-2] , Channel[-3], kernal[-2], kernal[-2], kernal[-2])

    Gl3 = relu(batchnorm(conv(Gl2,w3,filter_shape = filter_shape, input_shape = input_shape, conv_mode = 'deconv'),g = g3, b = b3))

    input_shape =  (None , None,Convlayersize[-3],Convlayersize[-3],Convlayersize[-3])
    filter_shape = (Channel[-3] , Channel[-4], kernal[-3], kernal[-3], kernal[-3])

    Gl4 = relu(batchnorm(conv(Gl3,w4,filter_shape = filter_shape, input_shape = input_shape, conv_mode = 'deconv'),g = g4, b= b4))

    input_shape =  (None, None, Convlayersize[-4],Convlayersize[-4],Convlayersize[-4])
    filter_shape = (Channel[-4], Channel[-5], kernal[-4], kernal[-4], kernal[-4])


    GlX = sigmoid(conv(Gl4,wx,filter_shape = filter_shape, input_shape = input_shape, conv_mode = 'deconv'))
    return GlX


X = T.tensor5()
Z = T.matrix()

gX = gen(Z, *gen_params)

print 'COMPILING'
t = time()
# _train_g = theano.function([X, Z, Y], cost, updates=g_updates)
# _train_d = theano.function([X, Z, Y], cost, updates=d_updates)
_gen = theano.function([Z], gX)
print '%.2f seconds to compile theano functions'%(time()-t)



# trX, trY, ntrain = load_shapenet_train()
n = 10
nbatch = 10
rng = np.random.RandomState(int(time()))

# sample_ymb = floatX(np.asarray(np.eye(3)))

z_dist = scipy.io.loadmat('Z_dist_class2.mat')
z_mean = z_dist['mean']
z_mean = np.reshape(z_mean,(Nz,1))

z_std = z_dist['std']
z_std = np.reshape(z_std,(Nz,1))

def gen_z(z_dist,nbatch):
    ret = np.zeros((nbatch,Nz))
    for j in xrange(Nz):
        z_tmp = np_rng.normal(z_mean[j],z_std[j],nbatch)
        ret[:,j] = z_tmp
    # print ret
    return ret

try:
    os.mkdir('Gen_models%d'%objectNumber)
except:
    pass

for j in xrange(n/nbatch):
    sample_zmb = floatX(gen_z(z_dist,nbatch))
    samples = np.asarray(_gen(sample_zmb))
    for i in xrange(nbatch):
        io.savemat('Gen_models%d/Gen_example_%d.mat'%(objectNumber,nbatch*j+i),{'instance':samples[i,:,:,:],'Z':sample_zmb[i,:]})





# niter = 1
# niter_decay = 1

