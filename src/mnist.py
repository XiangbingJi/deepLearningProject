
from __future__ import print_function

import sys
import os
import time

import numpy as np
np.random.seed(1234)  # for reproducibility

# specifying the gpu to use
# import theano.sandbox.cuda
# theano.sandbox.cuda.use('gpu1') 
import theano
import theano.tensor as T

import lasagne

import cPickle as pickle
import gzip

import batch_norm
from binary_connect import LogisticRegression, HiddenLayer, myMLP
import binary_connect
from pylearn2.datasets.mnist import MNIST
from pylearn2.utils import serial
from collections import OrderedDict


def test_mnist(binary,stochastic,SGD,ADAM,Nesterov,decay):
    
    # SGD,ADAM or Nesterov momentum
    if SGD:
        print ("Using SGD")
    if ADAM:
        print ("Using ADAM")
    if Nesterov:
        print ("Using Nesterov")
    # BN parameters
    batch_size = 20
    print("batch_size = "+str(batch_size))
    # alpha is the exponential moving average factor
    alpha = .15
    print("alpha = "+str(alpha))
    epsilon = 1e-4
    print("epsilon = "+str(epsilon))
    
    # MLP parameters
    num_units = 1024
    print("num_units = "+str(num_units))
    n_hidden_layers = 2
    print("n_hidden_layers = "+str(n_hidden_layers))
    
    # Training parameters
    num_epochs = 250
    print("num_epochs = "+str(num_epochs))
    
    # Dropout parameters
    dropout_in = 0. # 0. means no dropout
    print("dropout_in = "+str(dropout_in))
    dropout_hidden = 0.
    print("dropout_hidden = "+str(dropout_hidden))
    
    # BinaryConnect
    binary = binary
    print("binary = "+str(binary))
    stochastic = stochastic
    print("stochastic = "+str(stochastic))
    # (-H,+H) are the two binary values
    # H = "Glorot"
    H = 1.
    print("H = "+str(H))
    # W_LR_scale = 1.    
    W_LR_scale = 1 # "Glorot" means we are using the coefficients from Glorot's paper
    print("W_LR_scale = "+str(W_LR_scale))
    
    # Decaying LR 
    if decay:
       LR_start = .05
    else:
       LR_start = .01
    print("LR_start = "+str(LR_start))
    #LR_fin = 0.000003
    #print("LR_fin = "+str(LR_fin))
    #LR_decay = (LR_fin/LR_start)**(1./num_epochs)
    if decay:
       LR_decay=0.995
    else:
       LR_decay=1
    print("LR_decay = "+str(LR_decay))
    # BTW, LR decay might good for the BN moving average...
    
    print('Loading MNIST dataset...')
    
    train_set = MNIST(which_set= 'train', start=0, stop = 50000, center = True)
    valid_set = MNIST(which_set= 'train', start=50000, stop = 60000, center = True)
    test_set = MNIST(which_set= 'test', center = True)
    
    # bc01 format
    # print train_set.X.shape
    train_set.X = train_set.X.reshape(-1, 1, 28, 28)
    valid_set.X = valid_set.X.reshape(-1, 1, 28, 28)
    test_set.X = test_set.X.reshape(-1, 1, 28, 28)
    
    # flatten targets
    train_set.y = np.hstack(train_set.y)
    valid_set.y = np.hstack(valid_set.y)
    test_set.y = np.hstack(test_set.y)
    
    # Onehot the targets
    train_set.y = np.float32(np.eye(10)[train_set.y])    
    valid_set.y = np.float32(np.eye(10)[valid_set.y])
    test_set.y = np.float32(np.eye(10)[test_set.y])
    
    # for hinge loss
    train_set.y = 2* train_set.y - 1.
    valid_set.y = 2* valid_set.y - 1.
    test_set.y = 2* test_set.y - 1.
    
    train_set.X = train_set.X.reshape(50000, 1*28*28)
    valid_set.X = valid_set.X.reshape(10000, 1*28*28)
    test_set.X = test_set.X.reshape(10000, 1*28*28)
   
    print('Building the MLP...') 
    
    # Prepare Theano variables for inputs and targets
    input = T.matrix('inputs')
    target = T.matrix('targets')
    LR = T.scalar('LR', dtype=theano.config.floatX)
    rng = np.random.RandomState(678)
    
    classifier = myMLP(
        rng=rng,
        input=input,
        n_in=1*28*28,
        n_hidden=num_units,
        n_hiddenLayers=n_hidden_layers,
        n_out=10,
        binary=binary,
        stochastic=stochastic
    )

    loss = classifier.get_loss(target)
    
    if binary:        
        gparams = [T.grad(loss, param) for param in classifier.wrt]
        if SGD:
            updates = lasagne.updates.sgd(loss_or_grads=gparams, params=classifier.params, learning_rate=LR)
        if ADAM:
            updates = lasagne.updates.adam(loss_or_grads=gparams, params=classifier.params, learning_rate=LR)
        if Nesterov:
            updates = lasagne.updates.nesterov_momentum(loss_or_grads=gparams, params=classifier.params, learning_rate=LR)
        # Clipping  
        updates = binary_connect.clipping_scaling(updates,classifier,W_LR_scale)   
        # binarize Wb
        updates= binary_connect.update_Wb(updates,classifier,binary,stochastic)
        
    else:
        updates = lasagne.updates.sgd(loss_or_grads=loss, params=classifier.params, learning_rate=LR)

    test_loss = classifier.get_loss(target)
    test_err=classifier.errors(target)
   
    
    # Compile a function performing a training step on a mini-batch (by giving the updates dictionary) 
    # and returning the corresponding training loss:
    train_fn = theano.function(
        [input,target, LR], 
        loss, 
        updates=updates,
    )

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function(
        [input, target], 
        [test_loss, test_err],    
    )

    print('Training...')
    
    binary_connect.train(
            train_fn,val_fn,
            batch_size,
            LR_start,LR_decay,
            num_epochs,
            train_set.X,train_set.y,
            valid_set.X,valid_set.y,
            test_set.X,test_set.y)
    
    # print("display histogram")
    
    # W = lasagne.layers.get_all_layers(mlp)[2].W.get_value()
    # print(W.shape)
    
    # histogram = np.histogram(W,bins=1000,range=(-1.1,1.1))
    # np.savetxt(str(dropout_hidden)+str(binary)+str(stochastic)+str(H)+"_hist0.csv", histogram[0], delimiter=",")
    # np.savetxt(str(dropout_hidden)+str(binary)+str(stochastic)+str(H)+"_hist1.csv", histogram[1], delimiter=",")
    
    # Optionally, you could now dump the network weights to a file like this:
    # np.savez('model.npz', lasagne.layers.get_all_param_values(network))
if __name__ == "__main__":
    #test_mnist(binary=True,stochastic=False,SGD=True,ADAM=False,Nesterov=False,decay=True)
    #test_mnist(binary=True,stochastic=False,SGD=False,ADAM=True,Nesterov=False,decay=True)
    #test_mnist(binary=True,stochastic=False,SGD=False,ADAM=False,Nesterov=True,decay=True)
    #test_mnist(binary=True,stochastic=False,SGD=True,ADAM=False,Nesterov=False,decay=False)
    #test_mnist(binary=True,stochastic=False,SGD=False,ADAM=True,Nesterov=False,decay=False)
    #test_mnist(binary=True,stochastic=False,SGD=False,ADAM=False,Nesterov=True,decay=False)
    test_mnist(binary=True,stochastic=True,SGD=True,ADAM=False,Nesterov=False,decay=True)
    test_mnist(binary=False,stochastic=False,SGD=True,ADAM=False,Nesterov=False,decay=True)
