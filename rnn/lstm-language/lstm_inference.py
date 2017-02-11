import sys

import mxnet as mx

sys.path.append("..")
from rnn_unroll import rnn_unroll

def lstm_inference_symbol(num_lstm_layer, input_size, 
    num_hidden, num_embed, num_label, dropout=0.):

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
    hidden = mx.sym.Embedding(
        data = data,
        input_dim = input_size,
        weight = embed_weight,
        output_dim = num_embed,
        name = 'embed'
    )
    # stack LSTM
    for i in range(num_lstm_layer):
        if i==0:
            dp_ratio = 0.
        else:
            dp_ratio = dropout
        next_state = lstm(
            num_hidden = num_hidden,
            indata = hidden,
            mask = mask,
            prev_state = last_states[i],
            param = param_cells[i],
            seqidx = 0,
            layeridx = i,
            dropout = dp_ratio
        )
        last_states[i] = next_state
        hidden = next_state.h
    # decoder
    if dropout > 0.:
        hidden = mx.sym.Dropout(data = hidden, p = dropout)
    fc = mx.sym.FullyConnected(
        data = hidden,
        num_hidden = num_label,
        weight = cls_weight,
        bias = cls_bias,
        name = 'pred'
    )
    sm = mx.sym.SoftmaxOutput(data = fc, name = 'softmax')
    output = [sm]
    for state in last_states:
        output.append(state.c)
        output.append(state.h)
    return mx.sym.Group(output)

class LSTMInferenceModel(object):
    def __init__(self, num_lstm_layer, input_size,
            num_hidden, num_embed, num_label,
            arg_params, ctx=mx.cpu(), dropout=0.):
        self.sym = rnn_unroll(
            num_layers = num_lstm_layer,
            seq_len = 1,
            input_size = input_size,
            num_hidden = num_hidden,
            num_embed = num_embed,
            num_label = num_label,
            ignore_label = -1,
            mode = 'lstm', 
            bi_directional = False,
            dropout = 0., 
            train = False
        )
        batch_size = 1
        init_c = [('ptb_l%d_init_c'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
        init_h = [('ptb_l%d_init_h'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
        
        data_shape = [("data", (batch_size, 1)), ("mask", (batch_size, 1))]

        input_shapes = dict(init_c + init_h + data_shape)
        self.executor = self.sym.simple_bind(ctx=mx.cpu(), **input_shapes)

        for key in self.executor.arg_dict.keys():
            if key in arg_params:
                #print key, arg_params[key].shape, self.executor.arg_dict[key].shape
                arg_params[key].copyto(self.executor.arg_dict[key])
   
        state_name = []
        for i in range(num_lstm_layer):
            state_name.append("ptb_l%d_init_h" % i)
            state_name.append("ptb_l%d_init_c" % i)

        self.states_dict = dict(zip(state_name, self.executor.outputs[1:]))
        self.input_arr = mx.nd.zeros(data_shape[0][1])

    def forward(self, input_data, input_mask):
        input_data.copyto(self.executor.arg_dict["data"])
        input_mask.copyto(self.executor.arg_dict["mask"])
        self.executor.forward()
        for key in self.states_dict.keys():
            self.states_dict[key].copyto(self.executor.arg_dict[key])
        prob = self.executor.outputs[0].asnumpy()
        return prob
