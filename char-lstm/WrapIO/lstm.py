# pylink:skip-file
import sys
import mxnet as mx
import numpy as np
from collections import namedtuple
import time
import math

LSTMState = namedtuple("LSTMState", ["c", "h"])
LSTMParam = namedtuple("LSTMParam", ["i2h_weight", "i2h_bias", "h2h_weight", "h2h_bias"])
LSTMModel = namedtuple("LSTMModel", ["rnn_exec", "symbol", "init_states", "last_states", "seq_data", "seq_labels", "seq_outputs", "param_blocks"])

def lstm(num_hidden, indata, mask, prev_state, param, seqidx, layeridx, dropout=0.):
    """LSTM Cell symbol"""
    if dropout > 0:
        indata = mx.sym.Dropout(data = indata, p = dropout)
    i2h = mx.sym.FullyConnected(data = indata, 
                                weight = param.i2h_weight,
                                bias = param.i2h_bias,
                                num_hidden = num_hidden * 4,
                                name = "t%d_l%d_i2h" % (seqidx, layeridx))
    h2h = mx.sym.FullyConnected(data = prev_state.h,
                                weight = param.h2h_weight,
                                bias = param.h2h_bias,
                                num_hidden = num_hidden * 4,
                                name = "t%d_l%d_h2h" % (seqidx, layeridx))
    gates = i2h + h2h
    slice_gates = mx.sym.SliceChannel(gates, num_outputs = 4,
                                      name = "t%d_l%d_slice" % (seqidx, layeridx))
    
    in_gate = mx.sym.Activation(slice_gates[0], act_type = "sigmoid")
    in_transform = mx.sym.Activation(slice_gates[1], act_type = "tanh")
    forget_gate = mx.sym.Activation(slice_gates[2], act_type = "sigmoid")
    out_gate = mx.sym.Activation(slice_gates[3], act_type = "sigmoid")
    next_c = (forget_gate * prev_state.c) + (in_gate * in_transform)
    next_h = out_gate * mx.sym.Activation(next_c, act_type = "tanh")
    # mask out the output
    next_c = mx.sym.element_mask(next_c, mask, name = "t%d_l%d_c" % (seqidx, layeridx))
    next_h = mx.sym.element_mask(next_h, mask, name = "t%d_l%d_h" % (seqidx, layeridx))
    return LSTMState(c = next_c, h = next_h)

# Multiple inputs and single output
def lstm_unroll(num_lstm_layer, seq_len, input_size, 
              num_hidden, num_embed, num_label, ignore_label = 0, dropout = 0.):
    
    embed_weight = mx.sym.Variable("embed_weight")
    cls_weight = mx.sym.Variable("cls_weight")
    cls_bias = mx.sym.Variable("cls_bias")
    param_cells = []
    last_states = []
    
    for i in range(num_lstm_layer):
        param_cells.append(LSTMParam(i2h_weight = mx.sym.Variable("l%d_i2h_weight" % i),
                                     i2h_bias = mx.sym.Variable("l%d_i2h_bias" % i),
                                     h2h_weight = mx.sym.Variable("l%d_h2h_weight" % i),
                                     h2h_bias = mx.sym.Variable("l%d_h2h_bias" % i)))
        state = LSTMState(c = mx.sym.Variable("l%d_init_c" % i),
                          h = mx.sym.Variable("l%d_init_h" % i))
        last_states.append(state)
    assert(len(last_states) == num_lstm_layer)
    
    # Input data
    data = mx.sym.Variable('data')
    mask = mx.sym.Variable('mask')
    label = mx.sym.Variable('softmax_label')
    embed = mx.sym.Embedding(data = data, input_dim = input_size, 
                             weight = embed_weight, output_dim = num_embed, 
                             name = 'embed')
    wordvec = mx.sym.SliceChannel(data = embed, num_outputs = seq_len, squeeze_axis = 1)
    maskvec = mx.sym.SliceChannel(data = mask, num_outputs = seq_len, squeeze_axis = 1)
    # unroll the network
    hidden_all = [] # all hidden states
    for seqidx in range(seq_len):
        hidden = wordvec[seqidx]
        
        #stack LSTM
        for i in range(num_lstm_layer):
            if i == 0:
                dp_ratio = 0.
            else:
                dp_ratio = dropout
            next_state = lstm(num_hidden, indata = hidden,
                              mask = maskvec[seqidx],
                              prev_state = last_states[i],
                              param = param_cells[i],
                              seqidx = seqidx, layeridx = i, dropout = dp_ratio)
            hidden = next_state.h
            last_states[i] = next_state
        # state decoder
        hidden_all.append(hidden)

    
    hidden_concat = mx.sym.Concat(*hidden_all, dim = 0)
    # if we want to have attention, add it here.
    pred = mx.sym.FullyConnected(data = hidden_concat, num_hidden = num_label, weight = cls_weight, bias = cls_bias, name = 'pred')
    pred = mx.sym.Reshape(data = pred, shape = (-1, seq_len, num_label))
    #label = mx.sym.transpose(data = label)
    #label = mx.sym.Reshape(data = label, shape = (-1, ))
    sm = mx.sym.SoftmaxOutput(data = pred, label = label, ignore_label = ignore_label, name = 'softmax')
    
    return sm

# record the hidden states for sequence to sequence learning
def lstm_unroll_with_state(num_lstm_layer, seq_len, input_size,
                           num_hidden, num_embed, num_label, ignore_label=0, dropout=0.):
    
    # For weight we will share over whole network, we use ```mx.sym.Variable``` to represent it
    embed_weight = mx.sym.Variable("embed_weight") # embedding lookup table
    cls_weight = mx.sym.Variable("cls_weight") # classifier weight
    cls_bias = mx.sym.Variable("cls_bias") # classifier bias
    # Vertical initalization states and weights for LSTM unit
    param_cells = []
    last_states = []
    for i in range(num_lstm_layer):
        param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable("l%d_i2h_weight" % i),
                                     i2h_bias=mx.sym.Variable("l%d_i2h_bias" % i),
                                     h2h_weight=mx.sym.Variable("l%d_h2h_weight" % i),
                                     h2h_bias=mx.sym.Variable("l%d_h2h_bias" % i)))
        state = LSTMState(c=mx.sym.Variable("l%d_init_c" % i),
                          h=mx.sym.Variable("l%d_init_h" % i))
        last_states.append(state)
    assert(len(last_states) == num_lstm_layer)

    # Input data
    data = mx.sym.Variable('data') # input data, shape (batch, seq_length)
    mask = mx.sym.Variable('mask') # input mask, shape (batch, seq_length)
    label = mx.sym.Variable('softmax_label') # labels, shape (batch, seq_length)
    # Embedding calculation
    # We take the input and get all the embedding once
    # Which means the output will be in shape (batch, seq_length, output_embedding_dim)
    # Then we slice it will ```seq_len``` output
    # Which means seq_len output symbol, each's output shape is (batch, output_embedding_dim)
    embed = mx.sym.Embedding(data=data, input_dim=input_size,
                             weight=embed_weight, output_dim=num_embed, name='embed')
    wordvec = mx.sym.SliceChannel(data=embed, num_outputs=seq_len, squeeze_axis=1)
    maskvec = mx.sym.SliceChannel(data=mask, num_outputs=seq_len, squeeze_axis=1)

    # Now we can unroll the network
    hidden_all = []
    for seqidx in range(seq_len):
        hidden = wordvec[seqidx] # input to LSTM cell, comes from embedding

        # stack LSTM
        for i in range(num_lstm_layer):
            next_state = lstm(num_hidden, indata=hidden,
                              mask=maskvec[seqidx],
                              prev_state=last_states[i],
                              param=param_cells[i],
                              seqidx=seqidx, layeridx=i, dropout=dropout)
            hidden = next_state.h
            last_states[i] = next_state
        # decoder
        hidden_all.append(hidden) # last output of stack LSTM units

    hidden_concat = mx.sym.Concat(*hidden_all, dim=0)
    # If we want to have attention, add it here.
    pred = mx.sym.FullyConnected(data=hidden_concat, num_hidden=num_label,
                                 weight=cls_weight, bias=cls_bias, name='pred')


    label = mx.sym.transpose(data=label)
    label = mx.sym.Reshape(data=label, target_shape=(0,))

    sm = mx.sym.SoftmaxOutput(data=pred, label=label, ignore_label=ignore_label, name='softmax')

    outputs = [sm]
    # In the input we use init_c + init_h, so we will keep output in same convention
    for i in range(num_lstm_layer):
        state = last_states[i]
        outputs.append(mx.sym.BlockGrad(state.c, name="layer_%d_c" % i)) # stop back prop for last state
    for i in range(num_lstm_layer):
        state = last_states[i]
        outputs.append(mx.sym.BlockGrad(state.h, name="layer_%d_h" % i)) # stop back prop for last state
    return mx.sym.Group(outputs)

                          
def lstm_inference_symbol(num_lstm_layer, input_size, 
                          num_hidden, num_embed, num_label, dropout = 0.):
    seqidx = 0
    embed_weight = mx.sym.Variable("embed_weight")
    cls_weight = mx.sym.Variable("cls_weight")
    cls_bias = mx.sym.Variable("cls_bias")
    param_cells = []
    last_states = []

    for i in range(num_lstm_layer):
        param_cells.append(LSTMParam(i2h_weight = mx.sym.Variable("l%d_i2h_weight" % i),
                                     i2h_bias = mx.sym.Variable("l%d_i2h_bias" % i),
                                     h2h_weight = mx.sym.Variable("l%d_h2h_weight" % i),
                                     h2h_bias = mx.sym.Variable("l%d_h2h_bias" % i)))
        state = LSTMState(c = mx.sym.Variable("l%d_init_c" % i),
                          h = mx.sym.Variable("l%d_init_h" % i))
        last_states.append(state)
    assert(len(last_states) == num_lstm_layer)

    data = mx.sym.Variable('data') # input data, shape (batch, seq_length)
    mask = mx.sym.Variable('mask') # input mask, shape (batch, seq_length)
    hidden = mx.sym.Embedding(data=data, input_dim=input_size,
                             weight=embed_weight, output_dim=num_embed, name='embed')
    # stack LSTM
    for i in range(num_lstm_layer):
        if i == 0:
            dp = 0
        else:
            dp = dropout
        next_state = lstm(num_hidden, indata = hidden,
                          mask = mask,
                          prev_state = last_states[i],
                          param = param_cells[i],
                          seqidx = seqidx, layeridx = i,
                          dropout = dp)
        hidden = next_state.h
        last_states[i] = next_state
    #stack LSTM
    if dropout > 0:
        hidden = mx.sym.Dropout(data = hidden, p = dropout)
   
    fc = mx.sym.FullyConnected(data = hidden, weight = cls_weight, bias = cls_bias,
                               num_hidden = num_label)
    sm = mx.sym.SoftmaxOutput(data = fc, name = "sm")
    output = [sm]
    for state in last_states:
        output.append(state.c)
        output.append(state.h)
    return mx.sym.Group(output)

class LSTMInferenceModel(object):
    def __init__(self,
                 num_lstm_layer,
                 input_size,
                 num_hidden,
                 num_embed,
                 num_label,
                 arg_params,
                 ctx=mx.cpu(),
                 dropout=0.):
        self.sym = lstm_inference_symbol(
                                        num_lstm_layer,
                                         input_size,
                                         num_hidden,
                                         num_embed,
                                         num_label,
                                         dropout)
        batch_size = 1
        init_c = [('l%d_init_c'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
        init_h = [('l%d_init_h'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
        data_shape = [("data", (batch_size,))]
        mask_shape = [("mask", (batch_size,))]
        #label_shape =  [('softmax_label', (batch_size))]
        input_shapes = dict(init_c + init_h + data_shape + mask_shape)

        self.executor = self.sym.simple_bind(ctx=mx.cpu(), **input_shapes)
        for key in self.executor.arg_dict.keys():
            if key in arg_params:
                #print key, self.executor.arg_dict[key].shape, arg_params[key].shape
                arg_params[key].copyto(self.executor.arg_dict[key])

        state_name = []
        for i in range(num_lstm_layer):
            state_name.append("l%d_init_c" % i)
            state_name.append("l%d_init_h" % i)

        self.states_dict = dict(zip(state_name, self.executor.outputs[1:]))
        self.input_arr = mx.nd.zeros(data_shape[0][1])

    def forward(self, input_data, mask, new_seq=False):
        #print input_data.asnumpy()
        #print mask.asnumpy()
        if new_seq == True:
            for key in self.states_dict.keys():
                self.executor.arg_dict[key][:] = 0.
        input_data.copyto(self.executor.arg_dict["data"])
        mask.copyto(self.executor.arg_dict["mask"])
        self.executor.forward()
        for key in self.states_dict.keys():
            print self.states_dict[key].asnumpy().shape
            self.states_dict[key].copyto(self.executor.arg_dict[key])
        prob = self.executor.outputs[0].asnumpy()
        return prob

