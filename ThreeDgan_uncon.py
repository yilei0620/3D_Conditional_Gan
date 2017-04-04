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
tanh = activations.Tanh()


parameters = {'objectNumber': [3,8], 'Nz' : 200, 'Channel' :(1,64,128,256,512),  'kernal':(4,4,4,4), 'batchsize': 40, 'Convlayersize':(64,32,16,8,4), 'Genlrt' : 0.0021, 'Discrimlrt' : 0.000009 , 'beta' : 0.5, 'l2':2.5e-5, 'Genk' : 1 , 'niter_decay' : 1000}


for p in parameters:
    tmp = p + " = parameters[p]"
    exec(tmp)

Discrimlrt = Discrimlrt / (100.0/float(batchsize))
Genlrt = Genlrt / (100.0/float(batchsize))

# print conditional,type(batchsize),Channel[-1],kernal


gifn = inits.Normal(scale=0.002)
difn = inits.Normal(scale=0.002)
gain_ifn = inits.Normal(loc=1., scale=0.002)
bias_ifn = inits.Constant(c=0.)

temp = joblib.load('AutoEncmodels%s/75_dec_params.jl'%('_'.join([str(t) for t in objectNumber])))
gw1 = sharedX(temp[0],name = 'gw1')
gg1 = sharedX(temp[1],name = 'gg1')
gb1 = sharedX(temp[2],name = 'gb1')
gw2 = sharedX(temp[3],name = 'gw2')
gg2 = sharedX(temp[4],name = 'gg2')
gb2 = sharedX(temp[5],name = 'gb2')
gw3 = sharedX(temp[6],name = 'gw3')
gg3 = sharedX(temp[7],name = 'gg3')
gb3 = sharedX(temp[8],name = 'gb3')
gw4 = sharedX(temp[9],name = 'gw4')
gg4 = sharedX(temp[10],name = 'gg4')
gb4 = sharedX(temp[11],name = 'gb4')
gwx = sharedX(temp[12],name = 'gwx')

## filter_shape: (output channels, input channels, filter height, filter width, filter depth)

# gw1  = gifn((Nz, Channel[-1]*(Convlayersize[-1]**3)), 'gw1')
# gg1 = gain_ifn(Channel[-1]*(Convlayersize[-1]**3), 'gg1')
# gb1 = bias_ifn(Channel[-1]*(Convlayersize[-1]**3), 'gb1')
# gw2 = gifn((Channel[-1], Channel[-2], kernal[-1], kernal[-1], kernal[-1]), 'gw2')
# gg2 = gain_ifn((Channel[-2]), 'gg2')
# gb2 = bias_ifn((Channel[-2]), 'gb2')
# gw3 = gifn((Channel[-2], Channel[-3], kernal[-2], kernal[-2], kernal[-2]), 'gw3')
# gg3 = gain_ifn((Channel[-3]), 'gg3')
# gb3 = bias_ifn((Channel[-3]), 'gb3')
# gw4 = gifn((Channel[-3], Channel[-4], kernal[-3], kernal[-3], kernal[-3]), 'gw4')
# gg4 = gain_ifn((Channel[-4]), 'gg4')
# gb4 = bias_ifn((Channel[-4]), 'gb4')

# gwx = gifn((Channel[-4], Channel[-5], kernal[-4], kernal[-4], kernal[-4]), 'gwx')


dw1 = difn((Channel[1], Channel[0], kernal[0], kernal[0], kernal[0]), 'dw1')
dg1 = gain_ifn((Channel[1]), 'dg1')
db1 = bias_ifn((Channel[1]), 'db1')
dw2 = difn((Channel[2], Channel[1], kernal[1], kernal[1], kernal[1]), 'dw2')
dg2 = gain_ifn((Channel[2]), 'dg2')
db2 = bias_ifn((Channel[2]), 'db2')
dw3 = difn((Channel[3], Channel[2], kernal[2], kernal[2], kernal[2]), 'dw3')
dg3 = gain_ifn((Channel[3]), 'dg3')
db3 = bias_ifn((Channel[3]), 'db3')
dw4 = difn((Channel[4], Channel[3], kernal[3], kernal[3], kernal[3]), 'dw4')
dg4 = gain_ifn((Channel[4]), 'dg4')
db4 = bias_ifn((Channel[4]), 'db4')

dwy = difn((Channel[4]*(Convlayersize[4]**3), 1), 'dwy')


gen_params = [gw1, gg1, gb1, gw2, gg2, gb2, gw3, gg3, gb3, gw4 ,gg4, gb4, gwx]
discrim_params = [dw1, dg1, db1, dw2, dg2, db2, dw3, dg3, db3, dw4, dg4, db4, dwy]

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



def discrim(X, w1, g1, b1, w2, g2, b2, w3, g3, b3, w4, g4, b4, wy):

    filter_shape = (Channel[1] , Channel[0], kernal[0], kernal[0], kernal[0])
    Dl1 = lrelu(batchnorm(conv(X,w1,filter_shape = filter_shape),g = g1, b = b1))
 
    filter_shape = (Channel[2] , Channel[1], kernal[1], kernal[1], kernal[1])
    Dl2 = lrelu(batchnorm(conv(Dl1, w2,filter_shape = filter_shape), g = g2, b= b2))

    filter_shape = (Channel[3] , Channel[2], kernal[2], kernal[2], kernal[2])
    Dl3 = lrelu(batchnorm(conv(Dl2,w3,filter_shape = filter_shape), g = g3, b= b3))

    filter_shape = (Channel[4] , Channel[3], kernal[3], kernal[3], kernal[3])
    Dl4 = lrelu(batchnorm(conv(Dl3,w4,filter_shape = filter_shape), g = g4, b = b4))
    Dl4 = T.flatten(Dl4,2)
    DlY = sigmoid(T.dot(Dl4,wy))
    return DlY

X = T.tensor5()
Z = T.matrix()

gX = gen(Z, *gen_params)

p_real = discrim(X, *discrim_params)
p_gen = discrim(gX, *discrim_params)

d_cost_real = bce(p_real, T.ones(p_real.shape)).mean()
d_cost_gen = bce(p_gen, T.zeros(p_gen.shape)).mean()
g_cost_d = bce(p_gen, T.ones(p_gen.shape)).mean()


d_cost = d_cost_real + d_cost_gen
g_cost = g_cost_d

cost = [g_cost, d_cost, g_cost_d, d_cost_real, d_cost_gen]

Dlrt = sharedX(Discrimlrt)
Glrt = sharedX(Genlrt)
# Glrt2 = sharedX(Genlrt2)


d_updater = updates.Adam(lr=Dlrt, b1=0.5, regularizer=updates.Regularizer(l2=l2))
g_updater = updates.Adam(lr=Glrt, b1=0.5, regularizer=updates.Regularizer(l2=l2))
# g_updater2 = updates.Adam(lr=Glrt2, b1=0.5, regularizer=updates.Regularizer(l2=l2))
d_updates = d_updater(discrim_params, d_cost)
g_updates = g_updater(gen_params, g_cost)
# g_updates2 = g_updater2(gen_params, g_cost)
updates = d_updates + g_updates

print 'COMPILING'
t = time()
_train_g = theano.function([X, Z], cost, updates=g_updates)
# _train_g2 = theano.function([X, Z], cost, updates=g_updates2)
_train_d = theano.function([X, Z], cost, updates=d_updates)
# _gen = theano.function([Z], gX)
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
    os.mkdir('models%s'%('_'.join([str(t) for t in objectNumber])))
except:
    pass

f_log = open('models%s/%s_log.txt'%('_'.join([str(t) for t in objectNumber]),desc), 'wb')
log_fields = [
    'n_epochs', 
    'n_updates', 
    'n_examples', 
    'n_seconds',
    'g_cost',
    'd_cost',
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
Daccuracy = 1
# niter = 1
# niter_decay = 1

z_dist = scipy.io.loadmat('Z_dist_class_3_8.mat')
z_mean = z_dist['mean']
z_mean = np.reshape(z_mean,(Nz,1))

z_std = z_dist['std']
z_std = np.reshape(z_std,(Nz,1))

def gen_z(mean,std,nbatch):
    ret = np.zeros((nbatch,Nz))
    for j in xrange(Nz):
        z_tmp = np_rng.normal(mean[j],std[j],nbatch)
        ret[:,j] = z_tmp
    # print ret
    return ret

for epoch in range(1, niter_decay + 1):
    # trX, trY = shuffle(trX, trY)
    sIndex = np.arange(ntrain)
    np.random.shuffle(sIndex)
    for x_batch in iter_data(trX, shuffle_index = sIndex,size=batchsize, ndata = ntrain):
        # print x_batch.shape,x_batch.shape
        x_batch = floatX(np.reshape(x_batch,(x_batch.shape[0],1,64,64,64)))
        z_batch = floatX(gen_z(z_mean,z_std,x_batch.shape[0]))

        if Daccuracy > 0.44:
            cost = _train_d(x_batch, z_batch)
        cost = _train_g(x_batch, z_batch)
        
        Daccuracy = float(cost[1])
        # Daccuracy2 = float(cost[4])
        n_updates += 1
        n_examples += x_batch.shape[0]
        if n_updates%50 == 0:
            print 'epoch' + str(epoch),'time', str(time()-t)
            print 'g_cost %.4f d_cost %.4f g_cost_d %.4f, d_cost_real %.4f, d_cost_gen %.4f'%(float(cost[0]),float(cost[1]),float(cost[2]),float(cost[3]),float(cost[4]))
    n_epochs += 1

    Dlrt.set_value(floatX(Dlrt.get_value() - (Discrimlrt-Discrimlrt*0.01)/niter_decay))
    Glrt.set_value(floatX(Glrt.get_value() - (Genlrt - Genlrt*0.01)/niter_decay))

    g_cost = float(cost[0])
    d_cost = float(cost[1])
    log = [n_epochs, n_updates, n_examples, time()-t, g_cost, d_cost]
    print '%.0f %.4f %.4f'%(epoch, log[4], log[5])
    f_log.write(''.join([x+':'+str(y)+',' for x,y in zip(log_fields, log)] + ['\n']))
    f_log.flush()

    if n_epochs in [5, 10, 25, 50, 75, 100, 125, 150, 175, 200, 250, 300, 350, 400, 450, 500,600,700,800,900,1000]:
        joblib.dump([p.get_value() for p in gen_params], 'models%s/%d_gen_params.jl'%('_'.join([str(t) for t in objectNumber]), n_epochs))
        joblib.dump([p.get_value() for p in discrim_params], 'models%s/%d_discrim_params.jl'%('_'.join([str(t) for t in objectNumber]),n_epochs))





