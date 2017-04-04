import sys
sys.path.append('..')

import os
import json
from time import time
import numpy as np
from sklearn.externals import joblib
import scipy.io

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

from load_obj import load_shapenet_train, load_shapenet_test


relu = activations.Rectify()
sigmoid = activations.Sigmoid()
lrelu = activations.LeakyRectify()
bce = T.nnet.binary_crossentropy


parameters = {'objectNumber': [1,2,3,8], 'Nz' : 200, 'Channel' :(1,64,128,256,512),  'kernal':(4,4,4,4), 'batchsize': 50, 'Convlayersize':(64,32,16,8,4), 'lrate' : 0.001, 'beta' : 0.5, 'l2':2.5e-5, 'Genk' : 5 , 'niter_decay' : 200}


for p in parameters:
    tmp = p + " = parameters[p]"
    exec(tmp)


# print conditional,type(batchsize),Channel[-1],kernal


gifn = inits.Normal(scale=0.002)
difn = inits.Normal(scale=0.002)
gain_ifn = inits.Normal(loc=1., scale=0.002)
bias_ifn = inits.Constant(c=0.)


dw1  = gifn((Nz, Channel[-1]*(Convlayersize[-1]**3)), 'dw1')
dg1 = gain_ifn(Channel[-1]*(Convlayersize[-1]**3), 'dg1')
db1 = bias_ifn(Channel[-1]*(Convlayersize[-1]**3), 'db1')
dw2 = gifn((Channel[-1], Channel[-2], kernal[-1], kernal[-1], kernal[-1]), 'dw2')
dg2 = gain_ifn((Channel[-2]), 'dg2')
db2 = bias_ifn((Channel[-2]), 'db2')
dw3 = gifn((Channel[-2], Channel[-3], kernal[-2], kernal[-2], kernal[-2]), 'dw3')
dg3 = gain_ifn((Channel[-3]), 'dg3')
db3 = bias_ifn((Channel[-3]), 'db3')
dw4 = gifn((Channel[-3], Channel[-4], kernal[-3], kernal[-3], kernal[-3]), 'dw4')
dg4 = gain_ifn((Channel[-4]), 'dg4')
db4 = bias_ifn((Channel[-4]), 'db4')

dwx = gifn((Channel[-4], Channel[-5], kernal[-4], kernal[-4], kernal[-4]), 'dwx')


ew1 = difn((Channel[1], Channel[0], kernal[0], kernal[0], kernal[0]), 'ew1')
eg1 = gain_ifn((Channel[1]), 'eg1')
eb1 = bias_ifn((Channel[1]), 'eb1')
ew2 = difn((Channel[2], Channel[1], kernal[1], kernal[1], kernal[1]), 'ew2')
eg2 = gain_ifn((Channel[2]), 'eg2')
eb2 = bias_ifn((Channel[2]), 'eb2')
ew3 = difn((Channel[3], Channel[2], kernal[2], kernal[2], kernal[2]), 'ew3')
eg3 = gain_ifn((Channel[3]), 'eg3')
eb3 = bias_ifn((Channel[3]), 'eb3')
ew4 = difn((Channel[4], Channel[3], kernal[3], kernal[3], kernal[3]), 'ew4')
eg4 = gain_ifn((Channel[4]), 'eg4')
eb4 = bias_ifn((Channel[4]), 'eb4')

ewz = difn((Channel[4]*(Convlayersize[4]**3), Nz), 'dwz')

encode_params = [ew1, eg1, eb1, ew2, eg2, eb2, ew3, eg3, eb3, ew4 ,eg4, eb4, ewz]
decode_params = [dw1, dg1, db1, dw2, dg2, db2, dw3, dg3, db3, dw4, dg4, db4, dwx]

def decoder(Z, w1, g1, b1, w2, g2, b2, w3, g3, b3, w4, g4, b4, wx):
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


def encoder(X, w1, g1, b1, w2, g2, b2, w3, g3, b3, w4, g4, b4, wz):
    filter_shape = (Channel[1] , Channel[0], kernal[0], kernal[0], kernal[0])
    Dl1 = lrelu(batchnorm(conv(X,w1,filter_shape = filter_shape),g = g1, b = b1))

    filter_shape = (Channel[2] , Channel[1], kernal[1], kernal[1], kernal[1])
    Dl2 = lrelu(batchnorm(conv(Dl1, w2,filter_shape = filter_shape), g = g2, b= b2))

    filter_shape = (Channel[3] , Channel[2], kernal[2], kernal[2], kernal[2])
    Dl3 = lrelu(batchnorm(conv(Dl2,w3,filter_shape = filter_shape), g = g3, b= b3))

    filter_shape = (Channel[4] , Channel[3], kernal[3], kernal[3], kernal[3])
    Dl4 = lrelu(batchnorm(conv(Dl3,w4,filter_shape = filter_shape), g = g4, b = b4))
    Dl4 = T.flatten(Dl4,2)
    DlZ = sigmoid(T.dot(Dl4,wz))
    return DlZ


# def gen_Z(dist):
# 	mu = dist[:Nz]
# 	sigma = dist[Nz:]

X = T.tensor5()

encodeZ = encoder(X, *encode_params)
decodeX = decoder(encodeZ, *decode_params)



cost = bce(T.flatten(decodeX,2),T.flatten(X,2)).mean()

lrt = sharedX(lrate)
AutoEnc_parameter = encode_params + decode_params


updater = updates.Adam(lr=lrt, b1=0.8, regularizer=updates.Regularizer(l2=l2))
updates = updater(AutoEnc_parameter, cost)

print 'COMPILING'
t = time()
_train_ = theano.function([X], cost, updates=updates)
print '%.2f seconds to compile theano functions'%(time()-t)


mat = scipy.io.loadmat('models_stats.mat')
mat = mat['models']
num = np.array(mat[0][0][1])
names = mat[0][0][0][0]
objname = []
for j in range(len(objectNumber)):
    objname.append(names[objectNumber[j]][0])

desc = '3DshapeGan_' + '_'.join(objname)
try:
    os.mkdir('AutoEncmodels%s'%('_'.join([str(t) for t in objectNumber])))
except:
    pass
f_log = open('AutoEncmodels%s/%s_log.txt'%('_'.join([str(t) for t in objectNumber]),desc), 'wb')
log_fields = [
    'n_epochs', 
    'n_updates', 
    'n_examples', 
    'n_seconds',
    'cost'
]


for j in range(len(objectNumber)):
    if j == 0:
        trX, ntrain = load_shapenet_train(objectNumber[j])
    else:
        tmp, ntmp = load_shapenet_train(objectNumber[j])
        trX = np.concatenate((trX,tmp),axis = 0)
        ntrain += ntmp

print desc.upper() + ' Training!! ' 
n_updates = 0
n_check = 0
n_epochs = 0
n_updates = 0
n_examples = 0
t = time()

# niter = 1
# niter_decay = 1

for epoch in range(1, niter_decay + 1):
    # trX, trY = shuffle(trX, trY)
    sIndex = np.arange(ntrain)
    np.random.shuffle(sIndex)
    for x_batch in iter_data(trX, shuffle_index = sIndex,size=batchsize, ndata = ntrain):
        # print x_batch.shape,x_batch.shape
        x_batch = floatX(np.reshape(x_batch,(x_batch.shape[0],1,64,64,64)))
        cost = _train_(x_batch)
        
        n_updates += 1
        n_examples += x_batch.shape[0]
        if n_updates%50 == 0:
            print 'epoch' + str(epoch),'time', str(time()-t)
            print 'cost %.4f'%(float(cost))
    n_epochs += 1

    lrt.set_value(floatX(lrt.get_value() - lrate/niter_decay))


    log = [n_epochs, n_updates, n_examples, time()-t, float(cost)]
    print '%.0f %.4f'%(epoch, log[4])
    f_log.write(''.join([x+':'+str(y)+',' for x,y in zip(log_fields, log)] + ['\n']))
    f_log.flush()

    if n_epochs in [5, 10, 25, 50, 75, 100, 125, 150, 175, 200, 250, 300, 350, 400, 450, 500,600,700,800,900,1000]:
        joblib.dump([p.get_value() for p in encode_params], 'AutoEncmodels%s/%d_enc_params.jl'%('_'.join([str(t) for t in objectNumber]), n_epochs))
        joblib.dump([p.get_value() for p in decode_params], 'AutoEncmodels%s/%d_dec_params.jl'%('_'.join([str(t) for t in objectNumber]),n_epochs))











