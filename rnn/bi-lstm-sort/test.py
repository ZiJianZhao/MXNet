import mxnet as mx
import numpy as np

import re, os, sys, argparse, logging,collections
import codecs
sys.path.append('..')
from text_io import read_dict, get_data_label_text_id
from lstm_inference import BiLSTMInferenceModel

DEBUG = True

# ------------------------------ Process the data  ---------------------------------------

word2idx = read_dict('../../data/test/vocab.txt')

data_train, label_train = get_data_label_text_id('../../data/test/sorted_train.txt', word2idx)
data_valid, label_valid = get_data_label_text_id('../../data/test/sorted_valid.txt', word2idx)
data_test, label_test = get_data_label_text_id('../../data/test/sorted_test.txt', word2idx)

# ------------------------------- Parameter Defination -------------------------------

batch_size = 1

#  network parameters
num_lstm_layer = 2

input_size = len(word2idx)
dropout = 0.0

num_embed = 512
num_hidden = 512
num_label = len(word2idx)

# -------------------------------- BiLSTMInferenceModel -----------------------------------------
idx2word = {}
for k, v in word2idx.items():
	idx2word[v] = k

_, arg_params, __ = mx.model.load_checkpoint('%s/%s' % ('params', 'sort'), 5)

idx = 100
test_data = data_test[idx]
test_label = label_test[idx]
input_str = ""
for i in test_data:
	input_str += idx2word[i]
print "input sequence: ", input_str

output_str = ""
for i in test_label:
	output_str += idx2word[i]
print "output sequence: ", output_str

batch_size = 1
data_len = len(np.array(test_data))
data  = mx.nd.array(np.array(test_data).reshape(1, data_len))
mask = mx.nd.array(np.ones((data_len,)).reshape(1, data_len))

model = BiLSTMInferenceModel(
	seq_len = data_len, 
	input_size = input_size,
	num_hidden = num_hidden, 
	num_embed = num_embed,
	num_label = num_label, 
	arg_params = arg_params, 
	ctx=mx.cpu(), 
	dropout=0.0
)

prob = model.forward(data, mask)
model_str = ""
for i in xrange(data_len):
	model_str += idx2word[np.argmax(prob, axis = 1)[i]]
print "model sequence: ", model_str