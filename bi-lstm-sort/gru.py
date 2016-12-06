import mxnet as mx
from collections import namedtuple
GRUState = namedtuple("GRUState", ["h"])
GRUParam = namedtuple("GRUParam", ["gates_i2h_weight", "gates_i2h_bias",
								   "gates_h2h_weight", "gates_h2h_bias",
								   "trans_i2h_weight", "trans_i2h_bias",
								   "trans_h2h_weight", "trans_h2h_bias"])

def gru(num_hidden, indata, prev_state, param, seqidx, layeridx, dropout = 0.):
	if dropout > 0.:
		indata = mx.sym.Dropout(data = indata, p = dropout)
	i2h = mx.sym.FullyConnected(data = indata,
								weight = param.gates_i2h_weight,
								bias = param.gates_i2h_bias,
								num_hidden = num_hidden * 2,
								name = "t%d_l%d_gates_i2h")
	h2h = mx.sym.FullyConnected(data = indta,
								weight = param.gates_i2h_weight,
								bias = param.gates_i2h_bias,
								num_hidden = num_hidden * 2,
								name = "t%d_l%d_gates_h2h")
	gates = i2h + h2h
	slice_gates = mx.sym.SliceChannel(gates, 
									  num_outputs = 2,
									  name = "t%d_l%d_slice" % (seqidx, layeridx))
	update_gate = mx.sym.Activation(slice_gates[0], act_type = "sigmoid")
	reset_gate = mx.sym.Activation(slice_gates[1], act_type = "sigmoid")
	htrans_i2h = mx.sym.FullyConnected(data = indata, 
									   weight = param.tran_i2h_weight,
									   bias = param.trans_i2h_bais,
									   num_hidden = num_hidden,
									   name = "t%d_l%d_trans_i2h" % (seqidx, layeridx))
	h_after_reset = prev_state.h * reset_gate
	htrans_h2h = mx.sym.FullyConnected(data = h_after_reset,
									   weight = param.trans_h2h_weight,
									   bias = param.trans_h2h_bias,
									   num_hidden = num_hidden,
									   name = "t%d_l%d_trans_h2h" % (seqidx, layeridx))
	h_trans = htrans_i2h + htrans_h2h
	h_trans_active = mx.sym.Activation(h_trans, act_type = "tanh")
	next_h = prev_state.h + update_gate * (h_trans_active - prev_state.h)
	return GRUState(h = next_h)
 
