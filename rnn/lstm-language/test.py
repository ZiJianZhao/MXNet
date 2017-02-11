import mxnet as mx
import numpy as np

import re, os, sys, argparse, logging,collections
import codecs
from text_io import read_dict
from lstm_inference import LSTMInferenceModel

DEBUG = True

# ------------------------------ Process the data  ---------------------------------------

data_dir = '/slfs1/users/zjz17/github/data/ptb_data/'
vocab_path = os.path.join(data_dir, 'ptb.vocab.txt')
train_path = os.path.join(data_dir, 'ptb.train.txt')
valid_path = os.path.join(data_dir, 'ptb.valid.txt')
test_path = os.path.join(data_dir, 'ptb.test.txt')

word2idx = read_dict(vocab_path)
ignore_label = word2idx.get('<pad>')

# ------------------------------- Parameter Defination -------------------------------

batch_size = 1

#  network parameters
num_lstm_layer = 1

input_size = len(word2idx)
dropout = 0.0

num_embed = 512
num_hidden = 1024
num_label = len(word2idx)

# -------------------------------- BiLSTMInferenceModel -----------------------------------------
_, arg_params, __ = mx.model.load_checkpoint('%s/%s' % ('params', 'obama'), 6)

model = LSTMInferenceModel(
    num_lstm_layer = num_lstm_layer, 
    input_size = input_size,
    num_hidden = num_hidden, 
    num_embed = num_embed,
    num_label = num_label, 
    arg_params = arg_params, 
    ctx = mx.cpu(), 
    dropout = dropout 
)

idx2word = {}
for k, v in word2idx.items():
    idx2word[v] = k

test_str = ['the', 'united', 'states']

seq_length = 600

data = mx.nd.zeros((1,1))
mask = mx.nd.ones((1,1))

for i in xrange(seq_length):
    if i < len(test_str):
        tmp = np.zeros((1,))
        if word2idx.get(test_str[i]) is None:
            tmp[0] = word2idx.get('<UNK>')
        else:
            tmp[0] = word2idx.get(test_str[i])
        print '====================', data[0:1]
        data[0:1] = tmp[0]
        prob = model.forward(data, mask)
    else:
        tmp = np.zeros((1,))
        tmp[0] = word2idx.get(test_str[-1])

        data[0:1] = tmp[0]
        prob = model.forward(data, mask)
        prob = prob.reshape(-1,)
        prob = prob.argsort()[::-1]
        idx = 0
        while (prob[idx] == word2idx.get('<EOS>') 
            or prob[idx] == word2idx.get('<PAD>')):
            idx += 1
        next_char = idx2word[prob[idx]]
        test_str.append(next_char)

print test_str
