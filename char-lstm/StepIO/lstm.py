# pylint:skip-file
import sys
sys.path.insert(0, "../../python")
import mxnet as mx
import numpy as np
from collections import namedtuple
import time
import math

LSTMState = namedtuple("LSTMState", ["c", "h"])
LSTMParam = namedtuple("LSTMParam", ["i2h_weight", "i2h_bias",
										"h2h_weight", "h2h_bias"])
LSTMModel = namedtuple("LSTMModel", ["rnn_exec", "symbol",
										"init_states", "last_states",
										"seq_data", "seq_labels", "seq_outputs",
										"param_blocks"])

def lstm(num_hidden, indata, mask, 
	prev_state, param, seqidx, layeridx, dropout=0.):
	
	if dropout > 0.:
		indata = mx.sym.Dropout(data=indata, p=dropout)
	
	i2h = mx.sym.FullyConnected(
					data=indata,
					weight=param.i2h_weight,
					bias=param.i2h_bias,
					num_hidden=num_hidden * 4,
					name="t%d_l%d_i2h" % (seqidx, layeridx)
			)
	h2h = mx.sym.FullyConnected(
					data=prev_state.h,
					weight=param.h2h_weight,
					bias=param.h2h_bias,
					num_hidden=num_hidden * 4,
					name="t%d_l%d_h2h" % (seqidx, layeridx)
			)
	
	gates = i2h + h2h
	slice_gates = mx.sym.SliceChannel(
					gates,
					num_outputs=4,
					name="t%d_l%d_slice" % (seqidx, layeridx)
			)
	in_gate = mx.sym.Activation(slice_gates[0], act_type="sigmoid")
	in_transform = mx.sym.Activation(slice_gates[1], act_type="tanh")
	forget_gate = mx.sym.Activation(slice_gates[2], act_type="sigmoid")
	out_gate = mx.sym.Activation(slice_gates[3], act_type="sigmoid")
	next_c = (forget_gate * prev_state.c) + (in_gate * in_transform)
	next_h = out_gate * mx.sym.Activation(next_c, act_type="tanh")
	
	# mask out the output
	next_c = mx.sym.element_mask(next_c, mask, name = "t%d_l%d_c" % (seqidx, layeridx))
	next_h = mx.sym.element_mask(next_h, mask, name = "t%d_l%d_h" % (seqidx, layeridx))	
	return LSTMState(c=next_c, h=next_h)


# we define a new unrolling function here because the original
# one in lstm.py concats all the labels at the last layer together,
# making the mini-batch size of the label different from the data.
# I think the existing data-parallelization code need some modification
# to allow this situation to work properly
def lstm_unroll(num_lstm_layer, seq_len, input_size,
				num_hidden, num_embed, num_label, 
				ignore_label = 0, dropout=0.):
	
	embed_weight = mx.sym.Variable("embed_weight")
	cls_weight = mx.sym.Variable("cls_weight")
	cls_bias = mx.sym.Variable("cls_bias")
	param_cells = []
	last_states = []
	
	for i in range(num_lstm_layer):
		param_cells.append(
						LSTMParam(
							i2h_weight=mx.sym.Variable("l%d_i2h_weight" % i),
							i2h_bias=mx.sym.Variable("l%d_i2h_bias" % i),
							h2h_weight=mx.sym.Variable("l%d_h2h_weight" % i),
							h2h_bias=mx.sym.Variable("l%d_h2h_bias" % i)
							)
					)
		state = LSTMState(
						c=mx.sym.Variable("l%d_init_c" % i),
						h=mx.sym.Variable("l%d_init_h" % i)
					)
		last_states.append(state)

	assert(len(last_states) == num_lstm_layer)

	loss_all = []
	for seqidx in range(seq_len):
		# embeding layer
		data = mx.sym.Variable("data/%d" % seqidx)
		mask = mx.sym.Variable("mask/%d" % seqidx)
		hidden = mx.sym.Embedding(
						data=data,
						weight=embed_weight,
						input_dim=input_size,
						output_dim=num_embed,
						name="t%d_embed" % seqidx
					)

		# stack LSTM
		for i in range(num_lstm_layer):
			if i == 0:
				dp_ratio = 0.
			else:
				dp_ratio = dropout
			next_state = lstm(
						num_hidden = num_hidden,
						indata=hidden,
						mask = mask,
						prev_state=last_states[i],
						param=param_cells[i],
						seqidx=seqidx,
						layeridx=i,
						dropout=dp_ratio
				)
			hidden = next_state.h
			last_states[i] = next_state
		
		if dropout > 0.:
			hidden = mx.sym.Dropout(data=hidden, p=dropout)
		fc = mx.sym.FullyConnected(
						data=hidden,
						weight=cls_weight,
						bias=cls_bias,
						num_hidden=num_label
				)
		sm = mx.sym.SoftmaxOutput(
						data=fc,
						label=mx.sym.Variable('label/%d' % seqidx),
						name='t%d_sm' % seqidx
				)
		loss_all.append(sm)

	return mx.sym.Group(loss_all)


def lstm_inference_symbol(num_lstm_layer, input_size, num_hidden,
								num_embed, num_label, dropout=0.):
	seqidx = 0
	embed_weight=mx.sym.Variable("embed_weight")
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
		state = LSTMState(c=mx.sym.Variable("l%d_init_c" % i),
							h=mx.sym.Variable("l%d_init_h" % i))
		last_states.append(state)
	assert(len(last_states) == num_lstm_layer)

	data = mx.sym.Variable("data/%d" % seqidx)
	mask = mx.sym.Variable("mask/%d" % seqidx)
	hidden = mx.sym.Embedding(
						data=data,
						weight=embed_weight,
						input_dim=input_size,
						output_dim=num_embed,
						name="t%d_embed" % seqidx
	)
	# stack LSTM
	for i in range(num_lstm_layer):
		if i==0:
			dp=0.
		else:
			dp = dropout
		next_state = lstm(
						num_hidden = num_hidden,
						indata=hidden,
						mask = mask,
						prev_state=last_states[i],
						param=param_cells[i],
						seqidx=seqidx, 
						layeridx=i, 
						dropout=dp
		)
		hidden = next_state.h
		last_states[i] = next_state
	# decoder
	if dropout > 0.:
		hidden = mx.sym.Dropout(data=hidden, p=dropout)
	fc = mx.sym.FullyConnected(
					data=hidden, 
					weight=cls_weight, 
					bias=cls_bias,
					num_hidden=num_label
				)
	sm = mx.sym.SoftmaxOutput(
					data=fc, 
					label=mx.sym.Variable('label/%d' % seqidx),
					name='t%d_sm' % seqidx
				)
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
				dropout
		)

		batch_size = 1
		init_c = [('l%d_init_c'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
		init_h = [('l%d_init_h'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
		data_shape = [("data/0", (batch_size,))]
		mask_shape = [("mask/0", (batch_size,))]
		input_shapes = dict(init_c + init_h + data_shape + mask_shape)
		self.executor = self.sym.simple_bind(ctx=mx.cpu(), **input_shapes)

		for key in self.executor.arg_dict.keys():
			if key in arg_params:
				arg_params[key].copyto(self.executor.arg_dict[key])

		state_name = []
		for i in range(num_lstm_layer):
			state_name.append("l%d_init_c" % i)
			state_name.append("l%d_init_h" % i)

		self.states_dict = dict(zip(state_name, self.executor.outputs[1:]))
		self.input_arr = mx.nd.zeros(data_shape[0][1])

	def forward(self, input_data, new_seq=False):
		if new_seq == True:
			for key in self.states_dict.keys():
				self.executor.arg_dict[key][:] = 0.
		input_data.copyto(self.executor.arg_dict["data/0"])
		self.executor.forward()
		for key in self.states_dict.keys():
			self.states_dict[key].copyto(self.executor.arg_dict[key])
		prob = self.executor.outputs[0].asnumpy()
		return prob