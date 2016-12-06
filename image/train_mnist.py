import mxnet as mx

import numpy as np

import argparse
import os, sys 


def _download(data_dir):
    if not os.path.isdir(data_dir):
        os.system("mkdir " + data_dir)
    os.chdir(data_dir)
    if (not os.path.exists('train-images-idx3-ubyte')) or \
       (not os.path.exists('train-labels-idx1-ubyte')) or \
       (not os.path.exists('t10k-images-idx3-ubyte')) or \
       (not os.path.exists('t10k-labels-idx1-ubyte')):
       os.system("wget http://data.dmlc.ml/mxnet/data/mnist.zip")
       os.system("unzip -u mnist.zip; rm mnist.zip")
    os.chdir("..")

def get_loc(data, attr = {'lr_mult': '0.01'}):
    loc = mx.symbol.Convolution(data=data, num_filter = 30, kernel = (5, 5), stride = (2, 2))
    loc = mx.symbol.Activation(data = loc, act_type = 'relu')
    loc = mx.symbol.Pooling(data = loc, kernel = (2,2), stride = (2,2), pool_type = 'max')
    loc = mx.symbol.Convolution(data = loc, num_filter = 60, kernel = (3,3), stride = (1,1), pad = (1,1))
    loc = mx.symbol.Activation(data  = loc, act_type = 'relu')
    loc = mx.symbol.Pooling(data = loc, global_pool = True, kernel = (2,2), pool_type = 'avg')
    loc = mx.symbol.Flatten(data = loc)
    loc = mx.symbol.FullyConnected(data = loc, num_hidden = 6, name = 'stn_loc', attr = attr)
    return loc 

def get_lenet(add_stn = False):
    data = mx.symbol.Variable('data')
    if (add_stn):
        data = mx.sym.SpatialTransformer(
                                    data = data,
                                    loc = get_loc(data),
                                    target_shape = (28, 28),
                                    transform_type = "affine",
                                    sampler_type = "bilinear")
    conv1 = mx.symbol.Convolution(data = data, kernel = (5,5), num_filter = 20)
    tanh1 = mx.symbol.Activation(data = conv1, act_type = "tanh")
    pool1 = mx.symbol.Pooling(data = tanh1, kernel = (2,2), stride = (2,2), pool_type = "max")

    conv2 = mx.symbol.Convolution(data = pool1, kernel = (5,5), num_filter = 50)
    tanh2 = mx.symbol.Activation(data = conv2, act_type = 'tanh')
    pool2 = mx.symbol.Pooling(data = tanh2, pool_type = 'max', kernel = (2,2), stride = (2,2))

    flatten = mx.symbol.Flatten(data = pool2)
    fc1 = mx.symbol.FullyConnected(data = flatten, num_hidden = 500)
    tanh3 = mx.symbol.Activation(data = fc1, act_type = 'tanh')

    fc2 = mx.symbol.FullyConnected(data = tanh3, num_hidden = 10)
    lenet = mx.symbol.SoftmaxOutput(data = fc2, name = 'softmax')
    return lenet

def get_mlp():  
    data = mx.symbol.Variable('data')
    label = mx.symbol.Variable('softmax_label')
    fc1 = mx.symbol.FullyConnected(data = data, name = 'fc1', num_hidden = 128)
    act1 = mx.symbol.Activation(data = fc1, name = 'relu1', act_type = 'relu')
    fc2 = mx.symbol.FullyConnected(data = act1, name = 'fc2', num_hidden = 64)
    act2 = mx.symbol.Activation(data = fc2, name = 'relu2', act_type = 'relu')
    fc3 = mx.symbol.FullyConnected(data = act2, name = 'fc3', num_hidden = 10)
    mlp = mx.symbol.SoftmaxOutput(data = fc3, label = label, name = 'softmax')
    return mlp 
