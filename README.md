This final project was co-implemented by Jin Fang, Mengqing Wang, Shangshang Chen and Xiangbing Ji for ECBM6040. This research project is simply impossible without the help and support from Professor Aurel A. Lazar and all TAs. Especially, please allow us to dedicate our acknowledgment of gratitude towards all advisors and contributors.



We analyzed the paper and implemented the codes, which are structured as follows:

***src***



:   ***mnist.py***: main function for MLP



:   ***binary_connect.py***: classes definition for DNN layers applied in MNIST dataset



:   ***cifar10.py*** : main function for CNN



:   ***utils.py***: utility functions adapted from ECBM6040 hw3



:   ***nn.py***: classes definitions for DNN layers and CNN layers applied in CIFAR10 and SVHN datasets




***result***



:   ***plot***: graph plot for experiment results



:   ***raw***: experiment results



***README.md***

### Test error rates of DNNs trained on the MNIST, CIFAR-10 and SVHN


Experiments | Scripts | Results | Test Error
-------- | -------- | -------- | --------
Mnist, No Regularizer  | test_mnist(binary=False,stochastic=False,SGD=True,ADAM=False,Nesterov=False,decay=True)     | table2_mnist_No_regularizer.txt | 3.17%
Mnist,Binary,Deterministic  | test_mnist(binary=True,stochastic=False,SGD=True,ADAM=False,Nesterov=False,decay=True)     | table2_minist_binary_deterministic.txt | 2.77%
Mnist,Binary,Stochastic  | test_mnist(binary=True,stochastic=True,SGD=True,ADAM=False,Nesterov=False,decay=True)     | table2_mnist_binary_stochastic.txt | 4.85%
CIFAR10, No regularizer  | test_lenet(verbose=True,dataset='CIFAR10',binary=False,stochastic=False,learning_rate=0.1,n_epochs=1000,n_hidden=1000,n_hidden_layer=1,batch_size=200)     | result_cifar10.txt | 32.65%
CIFAR10, Binary, Deterministic  | test_lenet(verbose=True,dataset='CIFAR10',binary=True,stochastic=False,learning_rate=0.1,n_hidden=1000,n_epochs=1000,batch_size=200)   | det_cifar.txt | 39.77%
CIFAR10, Binary, Stochastic | test_lenet(verbose=True,dataset='CIFAR10',binary=True,stochastic=True,learning_rate=0.1,n_hidden=1000,n_epochs=1000,batch_size=200)   | sto_cifar.txt | 89.56%
SVHN with CNN,  No regularizer  | test_lenet(verbose=True,dataset=’SVHN’,binary=False,stochastic=False,learning_rate=0.1,n_epochs=1000,n_hidden=1000,n_hidden_layer=1,batch_size=200)    | result_svhn_nn1000.txt | 10.03%
SVHN with CNN, Binary, Deterministic | test_lenet(verbose=True,dataset='SVHN',binary=True,stochastic=False,learning_rate=0.04,n_hidden=500,n_epochs=1000,batch_size=100)   | det_svhn.txt| 80.40%
SVHN with CNN, Binary, Stochastic | test_lenet(verbose=True,dataset='SVHN',binary=True,stochastic=True,learning_rate=0.04,n_hidden=500,n_epochs=1000,batch_size=100)   | sto_svhn.txt | 80.47%
SVHN with MLP, Binary, Deterministic | test_mlp(verbose=True,dataset='SVHN',binary=True,stochastic=False,learning_rate=0.04,n_hidden=500,n_epochs=1000,batch_size=100)   | svhn_mlp.txt| 23.19%
