import sys
sys.path.append('..')

import os
import json
from time import time
import numpy as np
from sklearn.externals import joblib


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


parameters = {'conditional':True, 'Nz' : 200, 'Ny' : 40, 'Channel' :(1,64,128,256,512), 'full_conct':1024 , 'kernal':(4,4,4,4), 'batchsize':32, 'Convlayersize':(64,32,16,8,4), 'Genlrt' : 0.0025, 'Discrimlrt' : 1e-05 , 'beta' : 0.5, 'l2':2.5e-5, 'Genk' : 3 , 'niter': 50, 'niter_decay' : 50}


for p in parameters:
    tmp = p + " = parameters[p]"
    exec(tmp)

# print conditional,type(batchsize),Channel[-1],kernal


gifn = inits.Normal(scale=0.02)
difn = inits.Normal(scale=0.02)


## filter_shape: (output channels, input channels, filter height, filter width, filter depth)

gw1  = gifn((Nz+Ny, full_conct), 'gw1')
gw2 = gifn((full_conct+Ny, Channel[-1]*(Convlayersize[-1]**3)), 'gw2')
gw3 = gifn((Channel[-1] + Ny, Channel[-2], kernal[-1], kernal[-1], kernal[-1]), 'gw3')
gw4 = gifn((Channel[-2] + Ny, Channel[-3], kernal[-2], kernal[-2], kernal[-2]), 'gw4')
gw5 = gifn((Channel[-3] + Ny, Channel[-4], kernal[-3], kernal[-3], kernal[-3]), 'gw5')
gwx = gifn((Channel[-4] + Ny, Channel[-5], kernal[-4], kernal[-4], kernal[-4]), 'gwx')


dw1 = difn((Channel[1], Channel[0] + Ny, kernal[0], kernal[0], kernal[0]), 'dw1')
dw2 = difn((Channel[2], Channel[1] + Ny, kernal[1], kernal[1], kernal[1]), 'dw2')
dw3 = difn((Channel[3], Channel[2] + Ny, kernal[2], kernal[2], kernal[2]), 'dw3')
dw4 = difn((Channel[4], Channel[3] + Ny, kernal[3], kernal[3], kernal[3]), 'dw4')
dw5 = difn((Channel[4]*(Convlayersize[4]**3)+Ny, full_conct), 'dw5')
dwy = difn((full_conct+Ny, 1), 'dwy')


gen_params = [gw1, gw2, gw3, gw4, gw5, gwx]
discrim_params = [dw1, dw2, dw3, dw4, dw5, dwy]

def gen(Z, Y, w1, w2, w3, w4, w5, wx):
    yb = Y.dimshuffle(0, 1, 'x', 'x', 'x')
    Z = T.concatenate([Z, Y], axis=1)
    Gl1 = relu(T.dot(Z, w1))
    Gl1 = T.concatenate([Gl1,Y],axis = 1)

    Gl2 = relu(batchnorm(T.dot(Gl1,w2)))
    Gl2 = Gl2.reshape((Gl2.shape[0],Channel[-1],Convlayersize[-1],Convlayersize[-1],Convlayersize[-1]))
    Gl2 = conv_cond_concat(Gl2,yb)

    input_shape =  (batchsize,Channel[-1] + Ny,Convlayersize[-1],Convlayersize[-1],Convlayersize[-1])
    filter_shape = (Channel[-1] + Ny, Channel[-2], kernal[-1], kernal[-1], kernal[-1])

    Gl3 = relu(batchnorm(conv(Gl2,w3,filter_shape = filter_shape, input_shape = input_shape, conv_mode = 'deconv')))
    Gl3 = conv_cond_concat(Gl3,yb)

    input_shape =  (batchsize,Channel[-2] + Ny,Convlayersize[-2],Convlayersize[-2],Convlayersize[-2])
    filter_shape = (Channel[-2] + Ny, Channel[-3], kernal[-2], kernal[-2], kernal[-2])


    Gl4 = relu(batchnorm(conv(Gl3,w4,filter_shape = filter_shape, input_shape = input_shape, conv_mode = 'deconv')))
    Gl4 = conv_cond_concat(Gl4,yb)

    input_shape =  (batchsize,Channel[-3] + Ny,Convlayersize[-3],Convlayersize[-3],Convlayersize[-3])
    filter_shape = (Channel[-3] + Ny, Channel[-4], kernal[-3], kernal[-3], kernal[-3])


    Gl5 = relu(batchnorm(conv(Gl4,w5,filter_shape = filter_shape, input_shape = input_shape, conv_mode = 'deconv')))
    Gl5 = conv_cond_concat(Gl5,yb)

    input_shape =  (batchsize,Channel[-4] + Ny,Convlayersize[-4],Convlayersize[-4],Convlayersize[-4])
    filter_shape = (Channel[-4] + Ny, Channel[-5], kernal[-4], kernal[-4], kernal[-4])


    GlX = sigmoid(conv(Gl5,wx,filter_shape = filter_shape, input_shape = input_shape, conv_mode = 'deconv'))
    return GlX



def discrim(X, Y, w1, w2, w3, w4, w5, wy):
    yb = Y.dimshuffle(0, 1, 'x', 'x', 'x')
    X = conv_cond_concat(X,yb)

    # input_shape =  (batchsize,Channel[0] + Ny,Convlayersize[0],Convlayersize[0],Convlayersize[0])
    filter_shape = (Channel[1] , Channel[0]+ Ny, kernal[0], kernal[0], kernal[0])


    Dl1 = lrelu(conv(X,w1,filter_shape = filter_shape))
    Dl1 = conv_cond_concat(Dl1,yb)
 
    # input_shape =  (batchsize,Channel[1] + Ny,Convlayersize[1],Convlayersize[1],Convlayersize[1])
    filter_shape = (Channel[2] , Channel[1]+ Ny, kernal[1], kernal[1], kernal[1])


    Dl2 = lrelu(batchnorm(conv(Dl1,w2,filter_shape = filter_shape)))
    Dl2 = conv_cond_concat(Dl2,yb)

    # input_shape =  (batchsize,Channel[2] + Ny,Convlayersize[2],Convlayersize[2],Convlayersize[2])
    filter_shape = (Channel[3] , Channel[2]+ Ny, kernal[2], kernal[2], kernal[2])

    Dl3 = lrelu(batchnorm(conv(Dl2,w3,filter_shape = filter_shape)))
    Dl3 = conv_cond_concat(Dl3,yb)

    # input_shape =  (batchsize,Channel[3] + Ny,Convlayersize[3],Convlayersize[3],Convlayersize[3])
    filter_shape = (Channel[4] , Channel[3]+ Ny, kernal[3], kernal[3], kernal[3])

    Dl4 = lrelu(batchnorm(conv(Dl3,w4,filter_shape = filter_shape)))
    Dl4 = T.flatten(Dl4,2)
    Dl4 = T.concatenate([Dl4,Y], axis=1)

    Dl5 = lrelu(batchnorm(T.dot(Dl4, w5)))
    Dl5 = T.concatenate([Dl5, Y], axis=1)
    DlY = sigmoid(T.dot(Dl5,wy))
    return DlY

X = T.tensor5()
Z = T.matrix()
Y = T.matrix()

gX = gen(Z, Y, *gen_params)

p_real = discrim(X, Y, *discrim_params)
p_gen = discrim(gX, Y, *discrim_params)

d_cost_real = bce(p_real, T.ones(p_real.shape)).mean()
d_cost_gen = bce(p_gen, T.zeros(p_gen.shape)).mean()
g_cost_d = bce(p_gen, T.ones(p_gen.shape)).mean()


d_cost = d_cost_real + d_cost_gen
g_cost = g_cost_d

cost = [g_cost, d_cost, g_cost_d, d_cost_real, d_cost_gen]

Dlrt = sharedX(Discrimlrt)
Glrt = sharedX(Genlrt)


d_updater = updates.Adam(lr=Dlrt, b1=0.5, regularizer=updates.Regularizer(l2=l2))
g_updater = updates.Adam(lr=Glrt, b1=0.5, regularizer=updates.Regularizer(l2=l2))
d_updates = d_updater(discrim_params, d_cost)
g_updates = g_updater(gen_params, g_cost)
updates = d_updates + g_updates

print 'COMPILING'
t = time()
_train_g = theano.function([X, Z, Y], cost, updates=g_updates)
_train_d = theano.function([X, Z, Y], cost, updates=d_updates)
_gen = theano.function([Z, Y], gX)
print '%.2f seconds to compile theano functions'%(time()-t)

# tr_idxs = np.arange(len(trX))

desc = '3DshapeGan'
f_log = open('%s_log.txt'%desc, 'wb')
log_fields = [
    'n_epochs', 
    'n_updates', 
    'n_examples', 
    'n_seconds',
    'g_cost',
    'd_cost',
]



trX, trY, ntrain = load_shapenet_train()

print desc.upper() + ' Training!!'
n_updates = 0
n_check = 0
n_epochs = 0
n_updates = 0
n_examples = 0
t = time()

# niter = 1
# niter_decay = 1

for epoch in range(1, niter+ niter_decay + 1):
    # trX, trY = shuffle(trX, trY)
    sIndex = np.arange(ntrain)
    np.random.shuffle(sIndex)
    for x_batch, y_batch in iter_data(trX, trY, shuffle_index = sIndex,size=batchsize, ndata = ntrain):
        # print x_batch.shape,x_batch.shape
        x_batch = floatX(np.reshape(x_batch,(x_batch.shape[0],1,64,64,64)))
        z_batch = floatX(np_rng.uniform(0., 1., size=(x_batch.shape[0], Nz)))
        y_batch = floatX(y_batch)
        if n_updates % (Genk+1) == 0:
            cost = _train_d(x_batch, z_batch, y_batch)
        else:
            cost = _train_g(x_batch, z_batch, y_batch)
        n_updates += 1
        n_examples += x_batch.shape[0]
        if n_examples%9600 == 0:
            print 'epoch' + str(epoch),'time', str(time()-t)
    n_epochs += 1

    if n_epochs > niter:
        Dlrt.set_value(floatX(Dlrt.get_value() - Dlr/niter_decay))
        Glrt.set_value(floatX(Glrt.get_value() - Glr/niter_decay))

    g_cost = float(cost[0])
    d_cost = float(cost[1])
    log = [n_epochs, n_updates, n_examples, time()-t, g_cost, d_cost]
    print '%.0f %.4f %.4f'%(epoch, log[4], log[5])
    f_log.write(''.join([x+':'+str(y)+',' for x,y in zip(log_fields, log)] + ['\n']))
    f_log.flush()

    if n_epochs in [1,3, 5, 7,10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200]:
        joblib.dump([p.get_value() for p in gen_params], 'models/%d_gen_params.jl'% n_epochs)
        joblib.dump([p.get_value() for p in discrim_params], 'models/%d_discrim_params.jl'%n_epochs)





