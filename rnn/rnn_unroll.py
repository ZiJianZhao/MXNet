# pylint:skip-file
import sys
import mxnet as mx
import numpy as np
from collections import namedtuple
import time
import math

sys.path.append("..")
from rnn import RNN

def rnn_unroll(num_layers, seq_len, input_size,
        num_hidden, num_embed, num_label,
        ignore_label, mode = 'lstm', bi_directional = False,
        dropout = 0., train = True):

    # define weight variable and initial states
    embed_weight = mx.sym.Variable("embed_weight")
    cls_weight = mx.sym.Variable("cls_weight")
    cls_bias = mx.sym.Variable("cls_bias")

    # embedding layer
    data = mx.sym.Variable('data')
    mask = mx.sym.Variable('mask')

    embed = mx.sym.Embedding(
        data = data,
        input_dim = input_size,
        weight = embed_weight,
        output_dim = num_embed,
        name = 'embed'
    )

    rnn_outputs = RNN(
        data = embed, 
        mask = mask, 
        mode = mode, 
        seq_len = seq_len, 
        real_seq_len = None, 
        num_layers = num_layers, 
        num_hidden = num_hidden, 
        bi_directional = bi_directional, 
        states = None, 
        cells = None, 
        dropout = dropout, 
        name = 'ptb'
    ).get_outputs()

    hidden_all = []
    if bi_directional:
        for i in xrange(seq_len):
            hidden_all.append(
                mx.sym.Concat(*[rnn_outputs['last_layer'][2*i], rnn_outputs['last_layer'][2*i+1]], dim = 1)
            )
    else:    
        hidden_all = rnn_outputs['last_layer']

    # fullyconnected and softmax layer
    hidden_concat = mx.sym.Concat(*hidden_all, dim = 0)
    pred = mx.sym.FullyConnected(
        data = hidden_concat,
        num_hidden = num_label,
        weight = cls_weight,
        bias = cls_bias,
        name = 'pred'
    )
    if train:
        label = mx.sym.Variable('label')
        ## make label shape compatiable as hiddenconcat
        ## notice this reshape
        label = mx.sym.transpose(data = label)
        label = mx.sym.Reshape(data = label, shape = (-1,))
        ## notice the ignore label parameter
        sm = mx.sym.SoftmaxOutput(
            data = pred,
            label = label,
            ignore_label = ignore_label,
            use_ignore = True,
            name = 'softmax'
        )
        return sm
    else:
        sm = mx.sym.SoftmaxOutput(
            data = pred,
            name = 'softmax'
        )
        output = [sm]
        for i in range(num_layers):
            output.append(rnn_outputs['last_time'][2*i])
            output.append(rnn_outputs['last_time'][2*i+1])
        return mx.sym.Group(output)


