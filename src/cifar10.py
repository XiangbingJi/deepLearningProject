"""
Source Code for Homework 3.b of ECBM E6040, Spring 2016, Columbia University

Instructor: Prof. Aurel A. Lazar

This code is based on
[1] http://deeplearning.net/tutorial/logreg.html
[2] http://deeplearning.net/tutorial/mlp.html
[3] http://deeplearning.net/tutorial/lenet.html
"""
import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample

from utils import shared_dataset, load_data
from nn import LogisticRegression, HiddenLayer, myMLP, LeNetConvPoolLayer, train_nn, LeNetConvPoolLayer1

from pylearn2.datasets.mnist import MNIST
from pylearn2.datasets.zca_dataset import ZCA_Dataset
from pylearn2.utils import serial
import cPickle as pickle

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

def hard_sigmoid(x):
    return T.clip((x+1.)/2.,0,1)  



# TODO
def test_lenet(learning_rate=0.1, n_epochs=1000, nkerns=[16, 512], dataset='SVHN', binary=True,
               stochastic=False, batch_size=200, verbose=False,n_hidden=500):

    print ("learning rate: "+str(learning_rate))
    print ("n_epoches:     "+str(n_epochs))
    print ("batch_size:    "+str(batch_size))
    print ("dataset:       "+str(dataset))
    print ("nkerns:        "+str(nkerns[0])+", "+str(nkerns[1]))
    print ("binary:        "+str(binary))
    print ("stochastic     "+str(stochastic))
    
    # config: lr
    if (dataset == 'MNIST'):
        ##========================Mengqing's load_mnist==========================
        print("...Loading mnist")
        train_set = MNIST(which_set= 'train', start=0, stop = 50000, center = True)
        train_set_x = train_set.X.reshape(-1, 1, 28, 28)
        train_set_x = train_set_x.reshape(-1, 28*28)
        train_set_y = train_set.y.flatten()
        valid_set = MNIST(which_set= 'train', start=50000, stop = 60000, center = True)
        valid_set_x = valid_set.X.reshape(-1, 1, 28, 28)
        valid_set_x = valid_set_x.reshape(-1, 28*28)
        valid_set_y = valid_set.y.flatten()
        test_set = MNIST(which_set= 'test', center = True)
        test_set_x = test_set.X.reshape(-1, 1, 28, 28)
        test_set_x = test_set_x.reshape(-1, 28*28)
        test_set_y = test_set.y.flatten()
        train_set_x = theano.shared(numpy.asarray(train_set_x,dtype=theano.config.floatX),borrow=True)
        train_set_y = T.cast(theano.shared(numpy.asarray(train_set_y,dtype=theano.config.floatX),borrow=True), 'int32')
        valid_set_x = theano.shared(numpy.asarray(valid_set_x,dtype=theano.config.floatX),borrow=True)
        valid_set_y = T.cast(theano.shared(numpy.asarray(valid_set_y,dtype=theano.config.floatX),borrow=True), 'int32')
        test_set_x = theano.shared(numpy.asarray(test_set_x,dtype=theano.config.floatX),borrow=True)
        test_set_y = T.cast(theano.shared(numpy.asarray(test_set_y,dtype=theano.config.floatX),borrow=True), 'int32')
        n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size
        
        classifier_shape = 1*28*28
        ##========================================================================
    
    # config: lr=0.04, nn=500, epoch=1000, bs=100
    elif (dataset=='SVHN'): 
    
        ##===============================load SVHN ===========================
        print("...Loading SVHN hw3 whole set")
        datasets = load_data()
        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x, test_set_y = datasets[2]

        # compute number of minibatches for training, validation and testing
        n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size
        
        classifier_shape = 3*32*32
        ##===================================================================
    
    # config: lr=0.1, nn=1000, epoch=1000, bs=200
    else: 
        #=============================load cifar================================
        train_set_size = 45000
        print("train_set_size = "+str(train_set_size))

        print('...Loading CIFAR-10 dataset')

        preprocessor = serial.load("${PYLEARN2_DATA_PATH}/cifar10/pylearn2_gcn_whitened/preprocessor.pkl")
        train_set = ZCA_Dataset(
            preprocessed_dataset=serial.load("${PYLEARN2_DATA_PATH}/cifar10/pylearn2_gcn_whitened/train.pkl"), 
            preprocessor = preprocessor,
            start=0, stop = train_set_size)
        valid_set = ZCA_Dataset(
            preprocessed_dataset= serial.load("${PYLEARN2_DATA_PATH}/cifar10/pylearn2_gcn_whitened/train.pkl"), 
            preprocessor = preprocessor,
            start=45000, stop = 50000)  
        test_set = ZCA_Dataset(
            preprocessed_dataset= serial.load("${PYLEARN2_DATA_PATH}/cifar10/pylearn2_gcn_whitened/test.pkl"), 
            preprocessor = preprocessor)

        train_set_x = train_set.X.reshape(-1,3,32,32).reshape(-1,3*32*32)
        valid_set_x = valid_set.X.reshape(-1,3,32,32).reshape(-1,3*32*32)
        test_set_x = test_set.X.reshape(-1,3,32,32).reshape(-1,3*32*32)
        train_set_y = train_set.y.flatten()
        valid_set_y = valid_set.y.flatten()
        test_set_y = test_set.y.flatten()

        train_set_x = theano.shared(numpy.asarray(train_set_x,dtype=theano.config.floatX),borrow=True)
        train_set_y = T.cast(theano.shared(numpy.asarray(train_set_y,dtype=theano.config.floatX),borrow=True), 'int32')
        valid_set_x = theano.shared(numpy.asarray(valid_set_x,dtype=theano.config.floatX),borrow=True)
        valid_set_y = T.cast(theano.shared(numpy.asarray(valid_set_y,dtype=theano.config.floatX),borrow=True), 'int32')
        test_set_x = theano.shared(numpy.asarray(test_set_x,dtype=theano.config.floatX),borrow=True)
        test_set_y = T.cast(theano.shared(numpy.asarray(test_set_y,dtype=theano.config.floatX),borrow=True), 'int32')
        n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size   
        
        classifier_shape = 3*32*32
        #========================================================================

    rng = numpy.random.RandomState(23455)

    # train_set_size = 45000
    
    # preprocessor = serial.load("${PYLEARN2_DATA_PATH}/cifar10/pylearn2_gcn_whitened/preprocessor.pkl")
    # train_set = ZCA_Dataset(
    #     preprocessed_dataset=serial.load("${PYLEARN2_DATA_PATH}/cifar10/pylearn2_gcn_whitened/train.pkl"),
    #     preprocessor = preprocessor,
    #     start=0, stop = train_set_size)
    # valid_set = ZCA_Dataset(
    #     preprocessed_dataset= serial.load("${PYLEARN2_DATA_PATH}/cifar10/pylearn2_gcn_whitened/train.pkl"),
    #     preprocessor = preprocessor,
    #     start=45000, stop = 50000)
    # test_set = ZCA_Dataset(
    #     preprocessed_dataset= serial.load("${PYLEARN2_DATA_PATH}/cifar10/pylearn2_gcn_whitened/test.pkl"),
    #     preprocessor = preprocessor)

    # # bc01 format
    # # print train_set.X.shape
    # train_set.X = train_set.X.reshape(-1,3,32,32)
    # valid_set.X = valid_set.X.reshape(-1,3,32,32)
    # test_set.X = test_set.X.reshape(-1,3,32,32)
    
    # #train_set = MNIST(which_set= 'train', start=0, stop = 50000, center = True)
    # train_set_x = train_set.X.reshape(-1, 3, 32, 32)
    # train_set_x = train_set_x.reshape(-1, 32*32*3)
    # train_set_y = train_set.y.flatten()
    # #valid_set = MNIST(which_set= 'valid', start=50000, stop = 60000, center = True)
    # valid_set_x = valid_set.X.reshape(-1, 3, 32, 32)
    # valid_set_x = valid_set_x.reshape(-1, 32*32*3)
    # valid_set_y = valid_set.y.flatten()
    # #test_set = MNIST(which_set= 'test', center = True)
    # test_set_x = test_set.X.reshape(-1, 3, 32, 32)
    # test_set_x = test_set_x.reshape(-1, 32*32*3)
    # test_set_y = test_set.y.flatten()
    # print(test_set_y)
        
    # #train_set.X = train_set.X.reshape(-1, 1, 28, 28)
    # #valid_set.X = valid_set.X.reshape(-1, 1, 28, 28)
    # #test_set.X = test_set.X.reshape(-1, 1, 28, 28)
    
    # train_set_x = theano.shared(numpy.asarray(train_set_x,dtype=theano.config.floatX),borrow=True)
    # train_set_y = T.cast(theano.shared(numpy.asarray(train_set_y,dtype=theano.config.floatX),borrow=True), 'int32')
    # valid_set_x = theano.shared(numpy.asarray(valid_set_x,dtype=theano.config.floatX),borrow=True)
    # valid_set_y = T.cast(theano.shared(numpy.asarray(valid_set_y,dtype=theano.config.floatX),borrow=True), 'int32')
    # test_set_x = theano.shared(numpy.asarray(test_set_x,dtype=theano.config.floatX),borrow=True)
    # test_set_y = T.cast(theano.shared(numpy.asarray(test_set_y,dtype=theano.config.floatX),borrow=True), 'int32')
    
    # # compute number of minibatches for training, validation and testing
    # n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    # n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    # n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # Reshape matrix of rasterized images of shape (batch_size, 3 * 32 * 32)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size, 3, 32, 32))

    # TODO: Construct the first convolutional pooling layer
    # (32-5+1, 32-5+1)/2 = (14,14)
    layer0 = LeNetConvPoolLayer(
       rng,
       input=layer0_input,
       image_shape=(batch_size, 3, 32, 32),
       filter_shape=(nkerns[0], 3, 5, 5),
       poolsize=(2,2),
       binary=binary
    )

    # TODO: Construct the second convolutional pooling layer
    # (14-5+1,14-5+1)/2 = (5,5)
    layer1 = LeNetConvPoolLayer(
       rng,
       input=layer0.output,
       image_shape=(batch_size, nkerns[0], 14, 14),
       filter_shape=(nkerns[1], nkerns[0], 5, 5),
       poolsize=(2,2),
       binary=binary
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    layer2_input = layer1.output.flatten(2)

    # TODO: construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(
       rng,
       input=layer2_input,
       n_in=nkerns[1]*5*5,
       n_out=n_hidden,
       activation=T.tanh,
       binary=binary
    )

    # TODO: classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(
        input=layer2.output,
        n_in=n_hidden,
        n_out=10)

    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )


    

    updates = [] 
    if (binary):

        gparams_C1Wb = T.grad(cost, layer0.Wb)
        gparams_C1b = T.grad(cost, layer0.b)
        gparams_C2Wb = T.grad(cost, layer1.Wb)
        gparams_C2b = T.grad(cost, layer1.b)
        gparams_H1Wb = T.grad(cost, layer2.Wb)
        gparams_H1b = T.grad(cost, layer2.b)
        gparams_LW = T.grad(cost, layer3.W)
        gparams_Lb = T.grad(cost, layer3.b)

          
        # 1st layer
        updates.append((layer0.W,
                        T.clip(layer0.W - learning_rate * gparams_C1Wb,-1,1)))
        updates.append((layer0.b,
                        layer0.b - learning_rate * gparams_C1b))
        # 2nd layer
        updates.append((layer1.W,
                        T.clip(layer1.W - learning_rate * gparams_C2Wb,-1,1)))
        updates.append((layer1.b,
                        layer1.b - learning_rate * gparams_C2b))
        # 3rd layer
        updates.append((layer2.W,
                        T.clip(layer2.W - learning_rate * gparams_H1Wb,-1,1)))
        updates.append((layer2.b,
                        layer2.b - learning_rate * gparams_H1b))
        # 4th layer
        updates.append((layer3.W,
                        T.clip(layer3.W - learning_rate * gparams_LW,-1,1)))
        updates.append((layer3.b,
                        layer3.b - learning_rate * gparams_Lb))

        # updates Wb
        if (not stochastic):
            updates.append((layer0.Wb,
            T.cast(T.switch(T.round(T.clip(((layer0.W/1.)+1.)/2.,0,1)),1,-1), theano.config.floatX)))
            updates.append((layer1.Wb,
            T.cast(T.switch(T.round(T.clip(((layer1.W/1.)+1.)/2.,0,1)),1,-1), theano.config.floatX)))
            updates.append((layer2.Wb,
            T.cast(T.switch(T.round(T.clip(((layer2.W/1.)+1.)/2.,0,1)),1,-1), theano.config.floatX)))
        else:
            srng=RandomStreams(seed=234)
            updates.append((layer0.Wb,
            T.cast(T.switch(T.cast(srng.binomial(n=1,p=hard_sigmoid(layer0.W),size=T.shape(hard_sigmoid(layer0.W))), 
                theano.config.floatX),1,-1), theano.config.floatX)))

            updates.append((layer1.Wb,
            T.cast(T.switch(T.cast(srng.binomial(n=1,p=hard_sigmoid(layer1.W),size=T.shape(hard_sigmoid(layer1.W))), 
                theano.config.floatX),1,-1),theano.config.floatX)))

            updates.append((layer2.Wb,
            T.cast(T.switch(T.cast(srng.binomial(n=1,p=hard_sigmoid(layer2.W),size=T.shape(hard_sigmoid(layer2.W))), 
                theano.config.floatX),1,-1), theano.config.floatX)))



    else: 

        params = layer3.params + layer2.params + layer1.params + layer0.params
        grads = T.grad(cost, params)
        updates = [
            (param_i, param_i - learning_rate * grad_i)
            for param_i, grad_i in zip(params, grads)
        ]



    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    train_nn(train_model, validate_model, test_model,
        n_train_batches, n_valid_batches, n_test_batches, n_epochs, verbose)



    

if __name__ == "__main__":
    test_lenet(verbose=True,dataset='CIFAR10',binary=True,stochastic=False,learning_rate=0.1,n_hidden=1000,n_epochs=1000,batch_size=200)
   # test_lenet(verbose=True,dataset='SVHN',binary=True,stochastic=False,learning_rate=0.04,n_hidden=500,n_epochs=1000,batch_size=100)
    # test_convnet(verbose=True)
    # test_CDNN()
