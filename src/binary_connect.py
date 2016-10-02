import time

from collections import OrderedDict

import numpy

# specifying the gpu to use
# import theano.sandbox.cuda
# theano.sandbox.cuda.use('gpu1') 
import theano
import theano.tensor as T

import lasagne

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from copy import *
def hard_sigmoid(x):
    return T.clip((x+1.)/2.,0,1)

# The binarization function
def binarization(W,binary,stochastic,srng=None):
    print ("call binary")
    #W_copy = deepcopy(W)
    H=1
    # (deterministic == True) <-> test-time <-> inference-time
    if not binary:
        # print("not binary")
        Wb = W

    else:

        # [-1,1] -> [0,1]
        Wb = hard_sigmoid(W/H)

        # Stochastic BinaryConnect
        if stochastic:

            # print("stoch")
            Wb = T.cast(srng.binomial(n=1, p=Wb, size=T.shape(Wb)), theano.config.floatX)

        # Deterministic BinaryConnect (round to nearest)
        else:
            # print("det")
            Wb = T.round(Wb)

        # 0 or 1 -> -1 or 1
        Wb = T.cast(T.switch(Wb,H,-H), theano.config.floatX)

    return Wb


class BatchNormLayer(lasagne.layers.Layer):

    def __init__(self, input, n_in, n_out, epsilon=0.01, alpha=0.5):
        
        self.epsilon = epsilon
        self.alpha = alpha
        

        self.mean = theano.shared(value=0.0)
        self.std = theano.shared(value=1.0)

        self.beta = theano.shared(value=0.0, name='beta', borrow=True)
        self.gamma = theano.shared(value=1.0, name='gamma', borrow=True)

        self.mean = ((1 - self.alpha) * input.mean(keepdims=True) +
                           self.alpha * self.mean)
        self.std =  ((1 - self.alpha) * input.std(keepdims=True) +
                           self.alpha * self.std)

        normalized=(input - self.mean) * (self.gamma / self.std + self.epsilon) + self.beta

        self.output = normalized

class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out,binary=True,stochastic=True):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )
        
        self.Wb = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='Wb',
            borrow=True
        )

        if (binary):
            self.wrt = [self.Wb, self.b]
            self.p_y_given_x = T.nnet.softmax(T.dot(input, self.Wb) + self.b)
            self.output=T.dot(input, self.Wb) + self.b
        else:
            self.wrt = [self.W, self.b]
            self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
            self.output=self.p_y_given_x

        # parameters of the model

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyperplane for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of
        # hyperplane-k
        

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # keep track of model input
        self.input = input
        
        # parameters of the model
        self.params = [self.W,self.b]
        self.Ws=[self.W,self.Wb]

    def loss(self, target):
        # use the hinge loss function
        return T.mean(T.sqr(T.maximum(0.,1.-self.output*target)))

    def errors(self, target):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        return T.mean(T.neq(self.y_pred, T.argmax(target, axis=1)))

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh,binary=True,stochastic=False):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).
        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4


            W = theano.shared(value=W_values, name='W', borrow=True)
            #Wb = binarization(W,binary,stochastic)
            # print (W.get_value())
            # print (Wb.eval())

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b
        
        self.Wb = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='Wb',
            borrow=True
        )
     
        if (binary):
            lin_output = T.dot(input, self.Wb) + self.b
            self.wrt = [self.Wb, self.b]
        else:
            lin_output = T.dot(input, self.W) + self.b
            self.wrt = [self.W, self.b]
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W,self.b]
        self.Ws=[self.W,self.Wb]


class myMLP(object):
    """Multi-Layefr Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softmax layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, rng, input, n_in, n_hidden, n_out, n_hiddenLayers, binary, stochastic):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int or list of ints
        :param n_hidden: number of hidden units. If a list, it specifies the
        number of units in each hidden layers, and its length should equal to
        n_hiddenLayers.

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        :type n_hiddenLayers: int
        :param n_hiddenLayers: number of hidden layers
        """
        self.binary=binary
        self.stochastic=stochastic
  
        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a HiddenLayer with a tanh activation function connected to the
        # LogisticRegression layer; the activation function can be replaced by
        # sigmoid or any other nonlinear function.
        self.hiddenLayers = []
        self.normLayers=[]
        for i in xrange(n_hiddenLayers):
            h_input = input if i == 0 else self.hiddenLayers[i-1].output
            h_in = n_in if i == 0 else n_hidden

            # if binary==True, we append a binary hiddenlayer
            if binary==True:
                self.hiddenLayers.append(
                    HiddenLayer(
                        rng=rng,
                        input=h_input,
                        n_in=h_in,
                        n_out=n_hidden,
                        activation=T.tanh,
                        binary=True,
                        stochastic=stochastic
                    ))
                self.normLayers.append(
                    BatchNormLayer(
                        input=self.hiddenLayers[i].output,
                        n_in=n_hidden,
                        n_out=n_hidden
                    ))
            else:
                self.hiddenLayers.append(
                    HiddenLayer(
                        rng=rng,
                        input=h_input,
                        n_in=h_in,
                        n_out=n_hidden,
                        activation=T.tanh,
                        binary=False,
                        stochastic=False
                    ))

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayers[-1].output,
            n_in=n_hidden,
            n_out=n_out,
            binary=binary,
            stochastic=stochastic
        )
       
        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = sum([x.params for x in self.hiddenLayers], []) + self.logRegressionLayer.params
        self.wrt = sum([x.wrt for x in self.hiddenLayers], []) + self.logRegressionLayer.wrt
        self.Ws = sum([x.Ws for x in self.hiddenLayers], []) + self.logRegressionLayer.Ws
        # keep track of model input
        self.input = input
    
    def get_loss(self,target):
        return self.logRegressionLayer.loss(target)
    def get_output(self):
        return self.logRegressionLayer.output

# This functions clips the weights after the parameter update
def clipping_scaling(updates,network,W_LR_scale): 
    updates = OrderedDict(updates)      
    for param in network.params[0:6:2]:
            updates[param] = param + W_LR_scale*(updates[param] - param)
            updates[param] = T.clip(updates[param], -1,1)     
    return updates

def update_Wb(updates,network,binary,stochastic):  
    updates = OrderedDict(updates)   
    print len(network.Ws)
    for i in range(0,6,2):
            param=network.Ws[i+1]
            param1=network.Ws[i]
            #updates[param] =T.cast(T.switch(T.round(T.clip(((updates[param1]/1.)+1.)/2.,0,1)),1,-1), theano.config.floatX)
            updates[param]=binarization(updates[param1],binary,stochastic,srng = RandomStreams(seed=234))
    return updates
        
# Given a dataset and a model, this function trains the model on the dataset for several epochs
# (There is no default train function in Lasagne yet)
def train(train_fn,val_fn,
            batch_size,
            LR_start,LR_decay,
            num_epochs,
            X_train,y_train,
            X_val,y_val,
            X_test,y_test):
    
    # A function which shuffles a dataset
    def shuffle(X,y):
    
        shuffled_range = range(len(X))
        numpy.random.shuffle(shuffled_range)
        # print(shuffled_range[0:10])
        
        new_X = numpy.copy(X)
        new_y = numpy.copy(y)
        
        for i in range(len(X)):
            
            new_X[i] = X[shuffled_range[i]]
            new_y[i] = y[shuffled_range[i]]
            
        return new_X,new_y
    
    # This function trains the model a full epoch (on the whole dataset)
    def train_epoch(X,y,LR):
        
        loss = 0
        batches = len(X)/batch_size
        
        for i in range(batches):
            loss += train_fn(X[i*batch_size:(i+1)*batch_size],y[i*batch_size:(i+1)*batch_size],LR)
        
        loss/=batches
        
        return loss
    
    # This function tests the model a full epoch (on the whole dataset)
    def val_epoch(X,y):
        
        err = 0
        loss = 0
        batches = len(X)/batch_size
        
        for i in range(batches):
            new_loss, new_err = val_fn(X[i*batch_size:(i+1)*batch_size], y[i*batch_size:(i+1)*batch_size])
            err += new_err
            loss += new_loss
        
        err = err / batches * 100
        loss /= batches

        return err, loss
    
    # shuffle the train set
    X_train,y_train = shuffle(X_train,y_train)
    best_val_err = 100
    best_epoch = 1
    LR = LR_start
    
    # We iterate over epochs:
    for epoch in range(num_epochs):
        
        start_time = time.time()
        
        train_loss = train_epoch(X_train,y_train,LR)
        X_train,y_train = shuffle(X_train,y_train)
        
        val_err, val_loss = val_epoch(X_val,y_val)
        
        # test if validation error went down
        if val_err <= best_val_err:
            
            best_val_err = val_err
            best_epoch = epoch+1
            
            test_err, test_loss = val_epoch(X_test,y_test)
        
        epoch_duration = time.time() - start_time
        
        # Then we print the results for this epoch:
        print("Epoch "+str(epoch + 1)+" of "+str(num_epochs)+" took "+str(epoch_duration)+"s")
        print("  LR:                            "+str(LR))
        print("  training loss:                 "+str(train_loss))
        print("  validation loss:               "+str(val_loss))
        print("  validation error rate:         "+str(val_err)+"%")
        print("  best epoch:                    "+str(best_epoch))
        print("  best validation error rate:    "+str(best_val_err)+"%")
        print("  test loss:                     "+str(test_loss))
        print("  test error rate:               "+str(test_err)+"%") 
        
        # decay the LR
        LR *= LR_decay