import tensorflow as tf 
import os 
from keras.datasets import cifar10, cifar100
from keras.utils import to_categorical
import numpy as np 
import tensorflow.contrib.slim as slim 
import random
def load_cifar10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = normalization(x_train, x_test)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    return (x_train, y_train), (x_test, y_test)
def load_cifar100():
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    x_train, x_test = normalization(x_train, x_test)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    return (x_train, y_train), (x_test, y_test)
def normalization(x_train, x_test):
    x_train = x_train.astype('float32')/255.0
    x_test = x_test.astype('float32')/2550.0
    mean = np.mean(x_train, axis=(0,1,2,3))
    std = np.std(x_train, axis=(0,1,2,3))
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    return x_train, x_test  

def show_all_variables():
    trainable_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(trainable_vars, print_info=True)
def check_dir(logdir):
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    return logdir
def shuffle(x, y, seed=None):
    if seed == None:
        seed = np.random.randint(low=0,high=10000)
    np.random.seed(seed)
    np.random.shuffle(x)
    np.random.seed(seed)
    np.random.shuffle(y)
    return x, y