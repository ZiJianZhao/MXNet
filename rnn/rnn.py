#coding=utf-8
import mxnet as mx

class BaseRNN(object):
    '''A new design interface for RNN.

    Thinkings:
     1. no num_layers and bi_directional parameters, multi-layers rnn can be stacked by the user themselves.
     And bi_directional paramter is replaced by a additional paramter forward which means whether the forward
     direction is begin-to-end or end-to-begin.

    Args:

    '''
    def __init__(self, name='rnn'):
        super(BaseRNN, self).__init__()
        self.name = name
        self.reset()

    def __call__(self, data, states, mask=None):
        '''Construct symbol for one-step RNN'''
        raise NotImplementedError()

    @property
    def state_shape(self):
        '''Shape of hidden states'''
        raise NotImplementedError()

    def begin_state(self, func=mx.sym.zeros, **kwargs):
        '''Initial states for RNN'''
        states = []
        for shape in self.state_shape:
            self._init_counter = -1
            if shape is None:
                state = func(name = '%s_begin_state_%d' % (self.name, self._init_counter), **kwargs)
            else:
                state = func(name = '%s_begin_state_%d' % (self.name, self._init_counter), shape = shape, **kwargs)
            states.append(state)
        return states

    def reset(self):
        self._init_counter = -1
        self._step_counter = -1

    def unroll(self, data, seq_len, mask=None, begin_state=None, 
        dropout=0.0, forward=True, merge_outputs=False):

        self.reset()

        axis = 1
        if isinstance(data, mx.sym.Symbol):
            assert len(data.list_outputs()) == 1
            data = mx.sym.SliceChannel(
                data = data,
                num_outputs = seq_len,
                axis = axis,
                squeeze_axis = True,
                name = '%s_embed_slice_channel' % self.name
            )
            if mask is not None:
                mask = mx.sym.SliceChannel(
                    data = mask,
                    num_outputs = seq_len,
                    axis = axis,
                    squeeze_axis = True,
                    name = '%s_mask_slice_channel' % self.name
                )                
        else:
            assert len(data) == seq_len
        if begin_state is None:
            begin_state = self.begin_state()
        states = begin_state
        outputs = []
        for seqidx in range(seq_len):
            if forward:
                k = seqidx
            else:
                k = seq_len - seqidx - 1 
            if mask is not None:
                output, states = self(data[k], states, mask[k])
            else:
                output, states = self(data[k], states)
            if forward:
                outputs.append(output)
            else:
                outputs.insert(0, output)
        if merge_outputs:
            outputs = [mx.symbol.expand_dims(i, axis=axis) for i in outputs]
            outputs = mx.symbol.Concat(*outputs, dim=axis)
        return outputs, states

class LSTM(BaseRNN):
    def __init__(self, num_hidden, name='lstm'):
        super(LSTM, self).__init__(name = name)
        self.num_hidden = num_hidden
        self.i2h_weight = mx.sym.Variable('%s_i2h_weight' % self.name)
        self.i2h_bias = mx.sym.Variable('%s_i2h_bias' % self.name)
        self.h2h_weight = mx.sym.Variable('%s_h2h_weight' % self.name)
        self.h2h_bias = mx.sym.Variable('%s_h2h_bias' % self.name)

    @property
    def state_shape(self):
        return [(0, self.num_hidden), (0, self.num_hidden)]

    def __call__(self, data, states, mask=None):
        self._step_counter += 1
        name = '%s_t%d' % (self.name, self._step_counter)
        i2h = mx.sym.FullyConnected(
            data = data,
            weight = self.i2h_weight,
            bias = self.i2h_bias,
            num_hidden = self.num_hidden * 4,                    
            name = "%s_i2h" % name 
        )
        h2h = mx.sym.FullyConnected(
            data = states[0],
            weight = self.h2h_weight,
            bias = self.h2h_bias,
            num_hidden = self.num_hidden * 4,
            name = "%s_h2h" % name
        )
        gates = i2h + h2h
        slice_gates = mx.sym.SliceChannel(data = gates, num_outputs = 4)
        in_gate = mx.sym.Activation(slice_gates[0], act_type = "sigmoid")
        in_transform = mx.sym.Activation(slice_gates[1], act_type = "tanh")
        forget_gate = mx.sym.Activation(slice_gates[2], act_type = "sigmoid")
        out_gate = mx.sym.Activation(slice_gates[3], act_type = "sigmoid")
        next_c = (forget_gate * states[1]) + (in_gate * in_transform)
        next_h = out_gate * mx.sym.Activation(next_c, act_type = "tanh")
        if mask is not None:
            mask = mx.sym.Reshape(data = mask, shape = (-1, 1))
            next_h = (mx.sym.broadcast_mul(next_h, mask, name = 'next_h_broadcast_mul') + 
                mx.sym.broadcast_mul(states[0], 1 - mask, name = 'prev_h_broadcast_mul'))
            next_c = (mx.sym.broadcast_mul(next_c, mask, name = 'next_c_broadcast_mul') + 
                mx.sym.broadcast_mul(states[1], 1 - mask, name = 'prev_c_broadcast_mul'))
        
        return next_h, [next_h, next_c]


class GRU(BaseRNN):
    def __init__(self, num_hidden, name='gru'):
        super(GRU, self).__init__(name = name)
        self.num_hidden = num_hidden
        self.i2h_weight = mx.sym.Variable('%s_i2h_weight' % self.name)
        self.i2h_bias = mx.sym.Variable('%s_i2h_bias' % self.name)
        self.h2h_weight = mx.sym.Variable('%s_h2h_weight' % self.name)
        self.h2h_bias = mx.sym.Variable('%s_h2h_bias' % self.name)

    @property
    def state_shape(self):
        return [(0, self.num_hidden)]

    def __call__(self, data, states, mask=None):
        self._step_counter += 1
        name = '%s_t%d' % (self.name, self._step_counter)
        i2h = mx.sym.FullyConnected(
            data = data,
            weight = self.i2h_weight,
            bias = self.i2h_bias,
            num_hidden = self.num_hidden * 3,                    
            name = "%s_i2h" % name 
        )
        h2h = mx.sym.FullyConnected(
            data = states[0],
            weight = self.h2h_weight,
            bias = self.h2h_bias,
            num_hidden = self.num_hidden * 3,
            name = "%s_h2h" % name
        )
        i2h_z, i2h_r, i2h = mx.sym.SliceChannel(data = i2h, num_outputs = 3)
        h2h_z, h2h_r, h2h = mx.sym.SliceChannel(data = h2h, num_outputs = 3)
     
        update_gate = mx.sym.Activation(i2h_z + h2h_z, act_type = "sigmoid")
        reset_gate = mx.sym.Activation(i2h_r + h2h_r, act_type = "sigmoid")
        next_h_tmp = mx.sym.Activation(i2h + reset_gate * h2h, act_type="tanh")
        next_h = (1. - update_gate) * next_h_tmp + update_gate * states[0]

        if mask is not None:
            mask = mx.sym.Reshape(data = mask, shape = (-1, 1))
            next_h = (mx.sym.broadcast_mul(next_h, mask, name = 'next_h_broadcast_mul') + 
                mx.sym.broadcast_mul(states[0], 1 - mask, name = 'prev_h_broadcast_mul'))
        
        return next_h, [next_h]