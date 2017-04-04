import sys
sys.path.append('..')

import os
import json
from time import time
import numpy as np
from sklearn.externals import joblib
from scipy.io import loadmat
import scipy

read_path = 'visualization/Gen_models_AutoEnc3_8'
Nz = 200
nsample = 200
data = np.zeros((nsample,Nz))

for j in xrange(nsample):
	mat = loadmat('%s/Gen_example_%d.mat'%(read_path,j))
	z = mat['Z']
	data[j,:] = z

mdata = np.mean(data,axis = 0)
sdata = np.std(data,axis = 0)

scipy.io.savemat('Z_dist_class_3_8.mat',{'mean':mdata,'std':sdata})
scipy.io.savemat('Z_data_class_3_8.mat',{'lvector':data})