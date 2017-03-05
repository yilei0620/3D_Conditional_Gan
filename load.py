
import theano
from theano import tensor as T
import scipy.io
import numpy as np
from lib.data_utils import OneHot, shuffle, iter_data


def load_shapenet_train():
	theano.config.floatX = 'float32'

	mat = scipy.io.loadmat('models_stats.mat')
	mat = mat['models']
	num = np.array(mat[0][0][1])
	names = mat[0][0][0][0]
	print "# of classes:" + str(names.shape[0])

	Train_size = 9*np.sum(num[:,0])
	Test_size = 9*(np.sum(num[:,1])- np.sum(num[:,0]))
	print "Training instances: " + str(Train_size)
	print "Testing instances: " + str(Test_size)
	Train_size = 200
	Y_train = np.zeros((int(Train_size),names.shape[0]), dtype = 'int8')

	X_train = np.zeros((int(Train_size),64,64,64),dtype = 'int8')

	data_dir = "Voxel64_40"

	# training data phase
	indexing = 0
	for label,name in enumerate(names):
		print 'Loading ', data_dir,name[0]
		dirname = data_dir + '/' + name[0] + "/train"
		for i in xrange(1,int(num[label,0])+1):
			# if i >3:
			# 	break
			for j in xrange(1,10):
				filename = dirname + '/' + name[0] + '_' + '%04d'%i + '_' + str(j) + '.mat'
			#	print filename
				tmp = scipy.io.loadmat(str(filename))
				tmp = tmp['instance']
				X_train[indexing,:,:,:] = tmp
				Y_train[indexing,label] = 1.0
				indexing += 1
		# if label > 2:
		# 	break
	## shuffle the data
	# d = np.arange(indexing)
	# np.random.shuffle(d)
	# X_train = X_train[d,:,:,:]
	# Y_train = Y_train[d,:]
	# X_train = theano.shared(X_train,borrow = True)
	# Y_train = theano.shared(Y_train,borrow = True)


	return X_train,Y_train,indexing

def load_shapenet_test():
	theano.config.floatX = 'float32'

	mat = scipy.io.loadmat('models_stats.mat')
	mat = mat['models']
	num = np.array(mat[0][0][1])
	names = mat[0][0][0][0]
	print "# of classes:" + str(names.shape[0])
	Train_size = 9*np.sum(num[:,0])
	Test_size = 9*(np.sum(num[:,1])- np.sum(num[:,0]))
	print "Training instances: " + str(Train_size)
	print "Testing instances: " + str(Test_size)
	Y_test = np.zeros((int(Test_size),names.shape[0]), dtype = 'int8')

	X_test = np.zeros((int(Test_size),64,64,64),dtype = 'int8')

	data_dir = "Voxel64_40"

	# testing data phase
	indexing = 0
	for label,name in enumerate(names):
		print 'Loading ', data_dir,name[0]
		dirname = data_dir + '/' + name[0] + "/test"
		for i in xrange(int(num[label,0])+1,int(num[label,1])+1):
			# if i >100:
			#  	break
			for j in xrange(1,10):
				filename = dirname + '/' + name[0] + '_' + '%04d'%i + '_' + str(j) + '.mat'
			#	print filename
				tmp = scipy.io.loadmat(str(filename))
				tmp = tmp['instance']
				X_test[indexing,:,:,:] = tmp
				Y_test[indexing,label] = 1.0
				indexing += 1
		 # if label > 2:
		 # 	break
	## shuffle the data
	# d = np.arange(indexing)
	# np.random.shuffle(d)
	# X_test = X_test[d,:,:,:]
	# Y_test = Y_test[d,:]
	# X_test = theano.shared(X_test,borrow = True)
	# Y_test = theano.shared(Y_test,borrow = True)


	return X_test,Y_test,indexing

# X,Y = load_shapenet_train()
# trX, trY, ntrain = load_shapenet_test()
# niter = 1
# niter_decay = 1
# n_updates = 0

# sIndex = np.arange(ntrain)
# np.random.shuffle(sIndex)
# for epoch in range(1, 2):
#     # trX, trY = shuffle(trX, trY)
#     for imb, ymb in iter_data(trX, trY, shuffle_index=sIndex,size=100, ndata = ntrain):
#     	print imb.shape,ymb.shape
#     	imb = np.reshape(imb,(imb.shape[0],1,64,64,64))
#     	print imb.shape
#     	pass
#     n_updates += 1
    # n_examples += len(imb)




