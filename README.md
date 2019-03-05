# DenseNet-tensorflow
Simple tensorflow implementation of DenseNet for cifar-like dataset 
## **Description** 
This is a simple implementation of [Densely Connected Convolutional Neural Network](https://arxiv.org/abs/1608.06993) (DenseNet), written using tensorflow low-level API. To be simple, the model does not contain weight decay (l2 norm regularizations) and drop-out layer since it is believed that the inclusion of these into the network architecture would clearly improve the network performance. Also, the Adam Optimizer is used in replacement for Momentum Stochastic Gradient Descent (SGD). 

You could check out the training or testing process to see what is going on during the processes.
## **Dependencies**
This project has following dependencies:

Numpy `pip install numpy`

Keras `pip install keras`

Tensorflows `pip install tensorflow`
## **Usage**
### dataset 
cifar10 
cifar100 
### train
`python main.py --phase train --dataset cifar10 --lr 0.1`
### test 
`python main.py --phase test --dataset cifar10 --lr 0.1`
## Related work 
[DenseNet-cifar](https://github.com/giahuylhoang/DenseNet-keras/blob/master/DenseNet_cifar.py): if you're interested, you could check out my implementation of DenseNet using keras
