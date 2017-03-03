import mxnet as mx
import numpy as np

import argparse
import os, sys
import logging

sys.path.append("..")
from read_data import mnist_load
from basic_usage import init_logging

batch_size = 64

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


def get_mnist_iterator():
    train_images, valid_images, _ = mnist_load.load_data()
    train_images = train_images._replace(data = train_images.data.reshape((train_images.data.shape[0], -1)).astype(np.float32)/255)
    valid_images = valid_images._replace(data = valid_images.data.reshape((valid_images.data.shape[0], -1)).astype(np.float32)/255)
    train_iter = mx.io.NDArrayIter(train_images.data, train_images.label, batch_size=batch_size)
    test_iter = mx.io.NDArrayIter(valid_images.data, valid_images.label, batch_size=batch_size)
    return train_iter, test_iter

def model_training():
    mlp = get_mlp()
    train_iter, test_iter = get_mnist_iterator()
    model = mx.model.FeedForward(
        ctx = [mx.gpu(0), mx.gpu(1)],
        symbol = mlp,
        num_epoch = 10,
        learning_rate = 0.1,
        momentum = 0.9,
        wd = 0.00001
    )    
    model.fit(
        X = train_iter,
        eval_data = test_iter,
        batch_end_callback = mx.callback.Speedometer(batch_size, 100)
    )

def model_training_with_debugging():
    def norm_stat(d):
        return mx.nd.norm(d) / np.sqrt(d.size)
    mon = mx.mon.Monitor(
        interval = 100,
        stat_func = norm_stat,
        pattern = '.*weight',
        sort = True
    )
    mlp = get_mlp()
    train_iter, test_iter = get_mnist_iterator()
    model = mx.model.FeedForward(
        ctx = mx.gpu(0),
        symbol = mlp,
        num_epoch = 10,
        learning_rate = 0.1,
        momentum = 0.9,
        wd = 0.00001
    )    
    model.fit(
        X = train_iter,
        eval_data = test_iter,
        monitor = mon,
        batch_end_callback = mx.callback.Speedometer(batch_size, 100)
    )

def semi_customized_training():
    mlp = get_mlp()
    train_iter, test_iter = get_mnist_iterator()
    input_shapes = {'data':(batch_size, 28*28),'softmax_label':(batch_size, )}
    # executor, train executor which needs grad, attention on 'grad_req' 
    executor = mlp.simple_bind(
        ctx = mx.gpu(0),
        grad_req = 'write',
        **input_shapes
    )
    arg_arrays = dict(zip(mlp.list_arguments(), executor.arg_arrays))
    # initialization
    init = mx.init.Uniform(scale = 0.01)
    for name, arr in arg_arrays.items():
        if name not in input_shapes:
            init(name, arr)
    # optimizer definition
    opt = mx.optimizer.SGD(
        learning_rate = 0.1,
        momentum = 0.9,
        wd = 0.00001,
        rescale_grad = 1.0/batch_size
    )
    updater = mx.optimizer.get_updater(opt)
    # metric definition
    metric = mx.metric.Accuracy()
    # training
    data = arg_arrays[train_iter.provide_data[0][0]]
    label = arg_arrays[train_iter.provide_label[0][0]]
    for epoch in range(10):
        train_iter.reset()
        metric.reset()
        t = 0
        for batch in train_iter:
            # Copy data to executor input. Note the [:].
            data[:] = batch.data[0]
            label[:] = batch.label[0]
            
            # Forward
            executor.forward(is_train=True)
            
            # You perform operations on exe.outputs here if you need to.
            # For example, you can stack a CRF on top of a neural network.
            
            # Backward
            executor.backward()
            
            # Update
            for i, pair in enumerate(zip(executor.arg_arrays, executor.grad_arrays)):
                weight, grad = pair
                updater(i, grad, weight)
            metric.update(batch.label, executor.outputs)
            t += 1
            if t % 100 == 0:
                print 'epoch:', epoch, 'iter:', t, 'metric:', metric.get()

def customized_training():
    mlp = get_mlp()
    train_images, valid_images, _ = mnist_load.load_data()
    train_images = train_images._replace(data = train_images.data.reshape((train_images.data.shape[0], -1)).astype(np.float32)/255)
    valid_images = valid_images._replace(data = valid_images.data.reshape((valid_images.data.shape[0], -1)).astype(np.float32)/255)
    input_shapes = {'data':(batch_size, 28*28),'softmax_label':(batch_size, )}
    # executor, train executor which needs grad, attention on 'grad_req' 
    executor = mlp.simple_bind(ctx = mx.gpu(0),
                            grad_req = 'write',
                            **input_shapes)
    # initialization
    for r in executor.arg_arrays:
        r[:] = np.random.randn(*r.shape) * 0.01

    #training
    train_idx = range(train_images.data.shape[0])
    valid_idx = range(valid_images.data.shape[0])
    for epoch in xrange(10):
        print 'Starting epoch ', epoch
        np.random.shuffle(train_idx)
        for x in xrange(0, len(train_idx), batch_size):
            batch_data = train_images.data[train_idx[x:x+batch_size]]
            batch_label = train_images.label[train_idx[x:x+batch_size]]
            if batch_data.shape[0] != batch_size:
                continue
            executor.arg_dict['data'][:] = batch_data
            executor.arg_dict['softmax_label'][:] = batch_label
            executor.forward(is_train = True)
            executor.backward()
                # do weight updates in imperative
            for pname, W, G in zip(mlp.list_arguments(), executor.arg_arrays, executor.grad_arrays):
                # Don't update inputs
                # MXNet makes no distinction between weights and data.
                if pname in ['data', 'softmax_label']:
                    continue
                # what ever fancy update to modify the parameters
                W[:] = W - G * .001

        num_correct = 0
        num_total = 0
        for x in xrange(0, len(valid_images.data), batch_size):
            batch_data = valid_images.data[x:x+batch_size]
            batch_label = valid_images.label[x:x+batch_size]
            if batch_data.shape[0] != batch_size:
                continue
            executor.arg_dict['data'][:] = batch_data 
            executor.forward(is_train = False)
            num_correct += sum(batch_label == np.argmax(executor.outputs[0].asnumpy(), axis=1))
            num_total += len(batch_label)           
        print "Accuracy in valid data: ", num_correct / float(num_total)

def main():
    init_logging()
    model_training()

main()