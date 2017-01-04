import sys

import mxnet as mx

sys.path.append("..")
from lstm import *

def bi_lstm_inference_symbol(seq_len, input_size, num_hidden,
        num_embed, num_label, dropout = 0.):
    
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
    sm = mx.sym.SoftmaxOutput(
        data = pred,
        name = 'softmax'
    )
    output = [sm]
    for state in last_states:
        output.append(state.c)
        output.append(state.h)
    return mx.sym.Group(output)

class BiLSTMInferenceModel(object):
    def __init__(self, seq_len, input_size,
            num_hidden, num_embed, num_label,
            arg_params, ctx=mx.cpu(), dropout=0.):
        self.sym = bi_lstm_inference_symbol(
            seq_len = seq_len,
            input_size = input_size,
            num_hidden = num_hidden,
            num_embed = num_embed,
            num_label = num_label,
            dropout = dropout
        )
        batch_size = 1
        init_c = [('l%d_init_c'%l, (batch_size, num_hidden)) for l in range(2)]
        init_h = [('l%d_init_h'%l, (batch_size, num_hidden)) for l in range(2)]
        
        data_shape = [("data", (batch_size, seq_len, )), ("mask", (batch_size, seq_len,))]

        input_shapes = dict(init_c + init_h + data_shape)
        self.executor = self.sym.simple_bind(ctx=mx.cpu(), **input_shapes)

        for key in self.executor.arg_dict.keys():
            if key in arg_params:
                arg_params[key].copyto(self.executor.arg_dict[key])

        state_name = []
        for i in range(2):
            state_name.append("l%d_init_c" % i)
            state_name.append("l%d_init_h" % i)

        self.states_dict = dict(zip(state_name, self.executor.outputs[1:]))
        self.input_arr = mx.nd.zeros(data_shape[0][1])

    def forward(self, input_data, input_mask, new_seq=False):
        if new_seq == True:
            for key in self.states_dict.keys():
                self.executor.arg_dict[key][:] = 0.
        input_data.copyto(self.executor.arg_dict["data"])
        input_mask.copyto(self.executor.arg_dict["mask"])
        self.executor.forward()
        for key in self.states_dict.keys():
            self.states_dict[key].copyto(self.executor.arg_dict[key])
        prob = self.executor.outputs[0].asnumpy()
        return prob
