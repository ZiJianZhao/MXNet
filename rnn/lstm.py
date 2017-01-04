# pylint:skip-file
import sys
import mxnet as mx
import numpy as np
from collections import namedtuple
import time
import math
LSTMState = namedtuple("LSTMState", ["c", "h"])
LSTMParam = namedtuple("LSTMParam", ["i2h_weight", "i2h_bias",
                                     "h2h_weight", "h2h_bias"])
LSTMModel = namedtuple("LSTMModel", ["rnn_exec", "symbol",
                                     "init_states", "last_states", "forward_state", "backward_state",
                                     "seq_data", "seq_labels", "seq_outputs",
                                     "param_blocks"])

def lstm(num_hidden, indata, mask, prev_state, 
    param, seqidx, layeridx, dropout = 0.):

    if dropout > 0.:
        indata = mx.sym.Dropout(data = indata, p = dropout)
    i2h = mx.sym.FullyConnected(
        data = indata,
        weight = param.i2h_weight,
        bias = param.i2h_bias,
        num_hidden = num_hidden * 4,                    
        name = "t%d_l%d_i2h" % (seqidx, layeridx)
    )
    h2h = mx.sym.FullyConnected(
        data = prev_state.h,
        weight = param.h2h_weight,
        bias = param.h2h_bias,
        num_hidden = num_hidden * 4,
        name = "t%d_l%d_h2h" % (seqidx, layeridx)
    )
    gates = i2h + h2h
    slice_gates = mx.sym.SliceChannel(
        data = gates, 
        num_outputs = 4,
        name = "t%d_l%d_h2h" % (seqidx, layeridx)
    )
    in_gate = mx.sym.Activation(slice_gates[0], act_type = "sigmoid")
    in_transform = mx.sym.Activation(slice_gates[1], act_type = "tanh")
    forget_gate = mx.sym.Activation(slice_gates[2], act_type = "sigmoid")
    out_gate = mx.sym.Activation(slice_gates[3], act_type = "sigmoid")
    next_c = (forget_gate * prev_state.c) + (in_gate * in_transform)
    next_h = out_gate * mx.sym.Activation(next_c, act_type = "tanh")
    
    next_c = mx.sym.element_mask(next_c, mask, name = 'c_element_mask')
    next_h = mx.sym.element_mask(next_h, mask, name = 'h_element_mask')
    
    return LSTMState(c = next_c, h = next_h)


def lstm_unroll(num_lstm_layer, seq_len, input_size,
        num_hidden, num_embed, num_label,
        ignore_label, dropout = 0.):

    # define weight variable and initial states
    embed_weight = mx.sym.Variable("embed_weight")
    cls_weight = mx.sym.Variable("cls_weight")
    cls_bias = mx.sym.Variable("cls_bias")
    param_cells = []
    last_states = []
    for i in range(num_lstm_layer):
        param_cells.append(
            LSTMParam(
                i2h_weight = mx.sym.Variable("l%d_i2h_weight" % i),
                i2h_bias = mx.sym.Variable("l%d_i2h_bias" % i),
                h2h_weight = mx.sym.Variable("l%d_h2h_weight" % i),
                h2h_bias = mx.sym.Variable("l%d_h2h_bias" % i)
            )
        )
        last_states.append(
            LSTMState(
                c = mx.sym.Variable("l%d_init_c" % i),
                h = mx.sym.Variable("l%d_init_h" % i)
            )
        )
    assert len(last_states) == num_lstm_layer

    # embedding layer
    data = mx.sym.Variable('data')
    mask = mx.sym.Variable('mask')
    label = mx.sym.Variable('label')
    embed = mx.sym.Embedding(
        data = data,
        input_dim = input_size,
        weight = embed_weight,
        output_dim = num_embed,
        name = 'embed'
    )
    ## maybe some problemin this symbol, attention
    wordvec = mx.sym.SliceChannel(
        data = embed,
        num_outputs = seq_len,
        axis = 1,
        squeeze_axis = True
    )
    maskvec = mx.sym.SliceChannel(
        data = mask, 
        num_outputs = seq_len, 
        axis = 1,
        squeeze_axis = 1
    )
    # unrolled lstm
    hidden_all = []
    for seqidx in xrange(seq_len):
        hidden = wordvec[seqidx]
        ## stack lstm
        for i in xrange(num_lstm_layer):
            if i == 0:
                dp_ratio = 0.
            else:
                dp_ratio = dropout
            next_state = lstm(
                num_hidden = num_hidden,
                indata = hidden,
                mask = maskvec[seqidx],
                prev_state = last_states[i],
                param = param_cells[i],
                seqidx = seqidx,
                layeridx = i,
                dropout = dp_ratio
            )
            hidden = next_state.h
            last_states[i] = next_state
        ## output layer decoder
        if dropout > 0.:
            hidden = mx.sym.Dropout(data = hidden, p = dropout)
        hidden_all.append(hidden)

    # fullyconnected and softmax layer
    hidden_concat = mx.sym.Concat(*hidden_all, dim = 0)
    pred = mx.sym.FullyConnected(
        data = hidden_concat,
        num_hidden = num_label,
        weight = cls_weight,
        bias = cls_bias,
        name = 'pred'
    )
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


def bi_lstm_unroll(seq_len, input_size, num_hidden,
        num_embed, num_label, ignore_label = -1, dropout = 0.):
    
    embed_weight = mx.sym.Variable("embed_weight")
    cls_weight = mx.sym.Variable("cls_weight")
    cls_bias = mx.sym.Variable("cls_bias")
    last_states = []
    last_states.append(
        LSTMState(
            c = mx.sym.Variable("l0_init_c"),
            h = mx.sym.Variable("l0_init_h")
        )
    )
    last_states.append(
        LSTMState(
            c = mx.sym.Variable("l1_init_c"),
            h = mx.sym.Variable("l1_init_h")
        )
    )
    forward_param = LSTMParam(
        i2h_weight = mx.sym.Variable("l0_i2h_weight"),
        i2h_bias = mx.sym.Variable("l0_i2h_bias"),
        h2h_weight = mx.sym.Variable("l0_h2h_weight"),
        h2h_bias = mx.sym.Variable("l0_h2h_bias")
    )
    backward_param = LSTMParam(
        i2h_weight = mx.sym.Variable("l1_i2h_weight"),
        i2h_bias = mx.sym.Variable("l1_i2h_bias"),
        h2h_weight = mx.sym.Variable("l1_h2h_weight"),
        h2h_bias = mx.sym.Variable("l1_h2h_bias")
    )

    # embedding layer
    data = mx.sym.Variable("data")
    mask = mx.sym.Variable("mask")
    label = mx.sym.Variable("label")
    embed = mx.sym.Embedding(
        data = data,
        input_dim = input_size,
        weight = embed_weight,
        output_dim = num_embed,
        name = 'embed'
    )
    wordvec = mx.sym.SliceChannel(
        data = embed,
        num_outputs = seq_len,
        axis = 1,
        squeeze_axis = 1
    )
    maskvec = mx.sym.SliceChannel(
        data = mask, 
        num_outputs = seq_len, 
        axis = 1,
        squeeze_axis = 1
    )
    forward_hidden = []
    for seqidx in xrange(seq_len):
        hidden = wordvec[seqidx]
        next_state = lstm(
            num_hidden = num_hidden,
            indata = hidden,
            mask = maskvec[seqidx],
            prev_state = last_states[0],
            param = forward_param,
            seqidx = seqidx,
            layeridx = 0,
            dropout = dropout
        )
        hidden = next_state.h
        last_states[0] = next_state
        forward_hidden.append(hidden)
    backward_hidden = []
    for seqidx in xrange(seq_len):
        k = seq_len - seqidx - 1
        hidden = wordvec[k]
        next_state = lstm(
            num_hidden = num_hidden,
            indata = hidden,
            mask = maskvec[k],
            prev_state = last_states[1],
            param = backward_param,
            seqidx = k,
            layeridx = 1,
            dropout = dropout
        )
        hidden = next_state.h
        last_states[1] = next_state
        backward_hidden.insert(0, hidden)
    hidden_all = []
    for i in xrange(seq_len):
        hidden_all.append(
            mx.sym.Concat(*[forward_hidden[i], backward_hidden[i]], dim = 1)
        )
    hidden_concat = mx.sym.Concat(*hidden_all, dim = 0)

    pred = mx.sym.FullyConnected(
        data = hidden_concat,
        num_hidden = num_label,
        weight = cls_weight,
        bias = cls_bias,
        name = 'pred')
    label = mx.sym.transpose(data = label)
    label = mx.sym.Reshape(data = label, shape = (-1,))
    sm = mx.sym.SoftmaxOutput(
        data = pred,
        label = label,
        ignore_label = ignore_label,
        use_ignore = True,
        name = 'softmax'
    )
    return sm



