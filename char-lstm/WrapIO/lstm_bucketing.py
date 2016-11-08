from collections import namedtuple
import collections
import time
import math

import mxnet as mx
import numpy as np

from x_lstm import lstm_unroll_with_mul_outputs, lstm_unroll, LSTMInferenceModel

DUBUG = 0
'''
# ----------------- 1. Process the data  ---------------------------------------
def read_content(path):
	with open(path) as ins:
		content = ins.read()
		return content

def build_vocab(path):
	content = read_content(path)
	content = list(content)
	dic = {'PAD' : 0}
	idx = 1
	for word in content:
		if len(word) == 0:
			continue
		if not word in dic:
			dic[word] = idx 
			idx += 1
	return dic 

word2idx = build_vocab("./obama.txt")
vocab_size = len(idx2word)
if DEBUG:
	print "word2idx: ", word2idx

def get_text_id(path, word2idx):
	white_spaces = re.compile(r'[ \n\r\t]+')
	data = []
	index = 0
	with codecs.open(path, 'r', encoding='utf-8', errors='ignore') as fid:
		for line in fid:
			line = line.strip()
			line = white_spaces.split(line)
			if len(line) == 0:
				continue
			tmp = [word2idx.get(word) if word2idx.get(word) != None else word2idx.get('<UNK>') for word in line]
			data.append(tmp)
			index += 1
			if DEBUG:
				if index >= 100:
					return data
	return data 

# params
num_lstm_layer = 3
num_hidden = 512
num_embed = 256
batch_size = 32

num_epoch = 0
learning_rate = 0.01
momentum = 0.9
buckets = [129]
# state shape
init_c = [('l%d_init_c'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
init_h = [('l%d_init_h'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
init_states = init_c + init_h
state_names = [x[0] for x in init_states]

# Evaluation 
#def Perplexity(label, pred):
#	label = label.T.reshape((-1,))
	loss = 0.
	for i in range(pred.shape[0]):
		loss += -np.log(max(1e-10, pred[i][int(label[i])]))
#	return np.exp(loss / label.size)

def Perplexity(labels, preds):
	labels = labels.reshape((-1,))
	preds = preds.reshape((-1, preds.shape[2]))
	loss = 0.
	for i in range(preds.shape[0]):
		label = labels[i]
		if label > 0:
			loss += -np.log(max(1e-10, preds[i][int(label)]))
	return np.exp(loss / labels.size)

# symbolic generate function
def sym_gen(seq_len):
	sym = lstm_unroll(num_lstm_layer, seq_len, vocab_size,
						num_hidden=num_hidden, num_embed=num_embed,
						num_label=vocab_size)
	return sym


# bucketing execution module
# mod = mx.mod.BucketingModule(sym_gen, default_bucket_key=[10,20,30,40,50], context=mx.cpu())
#mod = mx.mod.BucketingModule(sym_gen, default_bucket_key=data_train.default_bucket_key, context=mx.gpu())

#mod.fit(data_train, num_epoch=1,
		eval_metric=mx.metric.np(Perplexity),
		batch_end_callback=mx.callback.Speedometer(batch_size, 50),
		initializer=mx.init.Xavier(factor_type="in", magnitude=2.34),
		optimizer='sgd',
		optimizer_params={'learning_rate':0.01, 'momentum': 0.9, 'wd': 0.00001})


data_train = BucketSentenceIter(path="./obama.txt",
								buckets=buckets, 
								vocab_size=vocab_size,
								batch_size=batch_size,
								init_states=init_states)
data_val = BucketSentenceIter(path="./obama.txt",
								buckets=buckets, 
								vocab_size=vocab_size,
								batch_size=batch_size,
								init_states=init_states)


model = mx.model.FeedForward(ctx=mx.cpu(),
							 symbol=sym_gen,
							 num_epoch=num_epoch,
							 learning_rate=learning_rate,
							 momentum=momentum,
							 wd=0.00001,
							 initializer=mx.init.Xavier(factor_type="in", magnitude=2.34))

import logging
head = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=head)
model.fit(X=data_train, eval_data=data_val,
			eval_metric = mx.metric.np(Perplexity),
			batch_end_callback=mx.callback.Speedometer(batch_size, 50),)

#model.save(prefix = "model")
'''

def read_content(path):
    with open(path) as ins:
        content = ins.read()
        return content

# Build a vocabulary of what char we have in the content
def build_vocab(path):
    content = read_content(path)
    content = list(content)
    idx = 1 # 0 is left for zero-padding
    the_vocab = {}
    for word in content:
        if len(word) == 0:
            continue
        if not word in the_vocab:
            the_vocab[word] = idx
            idx += 1
    return the_vocab

# We will assign each char with a special numerical id
def text2id(sentence, the_vocab):
    words = list(sentence)
    words = [the_vocab[w] for w in words if len(w) > 0]
    return words

def MakeRevertVocab(vocab):
    dic = {}
    for k, v in vocab.items():
        dic[v] = k
    return dic

vocab = build_vocab("./obama.txt")
#print vocab
vocab_size = len(vocab) + 1
revert_vocab = MakeRevertVocab(vocab)
import random
import bisect
# ----------------- generate words -------------------
def MakeInput(char, vocab, arr):
		idx = vocab[char]
		tmp = np.zeros((1,))
		tmp[0] = idx
		arr[:] = tmp

def _cdf(weights):
	total = sum(weights)
	result = []
	cumsum = 0
	for w in weights:
		cumsum += w
		result.append(cumsum / total)
	return result

def _choice(population, weights):
	assert len(population) == len(weights)
	cdf_vals = _cdf(weights)
	x = random.random()
	idx = bisect.bisect(cdf_vals, x)
	return population[idx]

def MakeOutput(prob, vocab, sample=False, temperature = 1.):
	if sample == False:
		idx = np.argmax(prob, axis=1)[0]
	else:
		fix_dict = [""] + [vocab[i] for i in range(1, len(vocab) + 1)]
		scale_prob = np.clip(prob, 1e-6, 1 - 1e-6)
		rescale = np.exp(np.log(scale_prob) / temperature)
		rescale[:] /= rescale.sum()
		return _choice(fix_dict, rescale[0, :])
	try:
		char = vocab[idx]
	except:
		char = ' '
	return char

_, arg_params, __ = mx.model.load_checkpoint("obama", 75)

gmodel = LSTMInferenceModel(
		num_lstm_layer = 3, 
		input_size = vocab_size,
		num_hidden=512, 
		num_embed=256,
		num_label=vocab_size, 
		arg_params=arg_params, 
		ctx=mx.cpu(), 
		dropout=0.2)

seq_length = 12
input_ndarray = mx.nd.zeros((1,))
mask_ndarray = mx.nd.ones((1,))
output = "The United States"
random_sample = False
new_sentence = True
ignore_length = len(output)

for i in range(seq_length):
	if i <= ignore_length - 1:
		MakeInput(output[i], vocab, input_ndarray)
	else:
		MakeInput(output[-1], vocab, input_ndarray)
	prob = gmodel.forward(input_ndarray, mask_ndarray, new_sentence)
	#print prob.shape
	new_sentence = False
	next_char = MakeOutput(prob, revert_vocab, random_sample)
	if next_char == ' ':
		new_sentence = True
	if i >= ignore_length - 1:
		output += next_char
print output