import mxnet as mx
import numpy as np

import argparse
import os, sys
import logging

# ==================================================
# ------------------- 1 logging---------------------
# logging levels: debug, info, warning, error, critical
# logging config: output to both file and terminal 
def init_logging(log_filename = 'LOG'):
    logging.basicConfig(
                    level    = logging.DEBUG,
                    format   = '%(filename)-20s LINE %(lineno)-4d %(levelname)-8s %(asctime)s %(message)s',
                    datefmt  = '%m-%d %H:%M:%S',
                    filename = log_filename,
                    filemode = 'w'
                )
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(filename)-20s LINE %(lineno)-4d %(levelname)-8s %(message)s');
    # tell the handler to use this format
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def test_logging():
    init_logging()
    logging.debug("This debug")
    logging.info("This info")
    logging.warning("This warning")
    logging.error("This error")
    logging.critical("This critical")

#test_logging()




def model():
    data = mx.symbol.Variable('data')
    fc = mx.symbol.FullyConnected(data = data, name = 'fc1', num_hidden = 128)
    symbol = mx.symbol.Activation(data = fc, name = 'relu1', act_type = 'relu')
    return symbol

# =====================================================
# ------------------- 2 basic usage--------------------
# 1.the network symbol shape: (batchsize, feature dimensions) 
# 2.the shape, values, gradient in the network all returned in the order of arguments, outputs, auxiliary states

# ------------------- 2.1 visualization ---------------
# the shape along the line omits the first dimension which is always batchsize
def visualization(symbol = None, shape = None):
    if symbol is None:
        return
    dot_graph = mx.viz.plot_network(
                        symbol = symbol,
                        save_format = 'pdf',
                        shape = shape,
                        node_attrs = {'shape':'rect', 'fixedsize':'false'},
                        hide_weights = True
                    )
    dot_graph.render('network.gv' ,view = True)


#visualization(symbol = model(), shape = {'data':(4,5)})

# =========================================================
# ------------------- 3 Symbol Manipulation ---------------
# Reference: http://mxnet.io/tutorials/python/symbol.html
# ------------------- 3.1 Infer shape ---------------------
def infer_shape(symbol = None, **shape):
    if symbol is None:
        return
    arg_name = symbol.list_arguments()
    out_name = symbol.list_outputs()
    arg_shape, out_shape, aux_shape = symbol.infer_shape(**shape)
    print {'input': dict(zip(arg_name, arg_shape)),
    'output': dict(zip(out_name, out_shape))}
    
#infer_shape(model(), **{'data':(3,4)})

# ------------------- 3.2 Bind data and Evaluate ---------------------
def bind_evaluate():
    a = mx.symbol.Variable("A")
    b = mx.symbol.Variable("B")
    c = a + b
    # allocate space for inputs
    input_arguments = {}
    input_arguments['A'] = mx.nd.ones((10, ), ctx=mx.cpu())
    input_arguments['B'] = mx.nd.ones((10, ), ctx=mx.cpu())
    # allocate space for gradients
    grad_arguments = {}
    grad_arguments['A'] = mx.nd.ones((10, ), ctx=mx.cpu())
    grad_arguments['B'] = mx.nd.ones((10, ), ctx=mx.cpu())
    ex = c.bind(ctx = mx.cpu(),
                args = input_arguments,
                args_grad = grad_arguments,
                grad_req = 'write'
            )
    ex.arg_dict['A'][:] = np.random.rand(10,)
    ex.arg_dict['B'][:] = np.random.rand(10,)
    ex.forward()
    out_grad = mx.nd.ones((10,))
    ex.backward([out_grad])
    print 'number of outputs = %d, first output = \n %s' % \
    (len(ex.outputs), ex.outputs[0].asnumpy())
    grad_arrays = dict(zip(c.list_arguments(), ex.grad_arrays))
    print 'grads:'
    for name, arr in grad_arrays.items():
        print name, arr.asnumpy()

#bind_evaluate()

# ------------------- 3.3 Auto Differention ---------------------
def auto_differention():
    a = mx.symbol.Variable("a")
    b = mx.symbol.Variable("b")
    c = a + b
    d = c.grad(wrt = ('a'))
    print 'arguments:', d.list_arguments()
    print 'outputs:', d.list_outputs()    
    ex = d.bind(ctx = mx.cpu(),
                args = {c.name+'_0_grad': mx.nd.ones([3,4]),
                        'a': mx.nd.ones([3,4])*2,
                        'b': mx.nd.ones([3,4])*3}
            )
    ex.forward()
    print 'output:\n', ex.outputs[0].asnumpy()

#auto_differention()