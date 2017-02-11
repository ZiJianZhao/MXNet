import mxnet as mx
import numpy as np

import re, os, sys, argparse, logging,collections
import codecs

sys.path.append("..")
from rnn_unroll import rnn_unroll
from text_io import read_dict, get_text_id
from sequence_iter import SequenceIter, DummyIter

def init_logging(log_filename = 'Log'):
    logging.basicConfig(
        level    = logging.DEBUG,
        format   = '%(filename)-20s LINE %(lineno)-4d %(levelname)-8s %(asctime)s %(message)s',
        datefmt  = '%m-%d %H:%M:%S',
        filename = log_filename,
        filemode = 'w'
    )
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(filename)-20s LINE %(lineno)-4d %(levelname)-8s %(message)s');
    # tell the handler to use this format
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

init_logging()

# ----------------- 1. Process the data  ---------------------------------------
data_dir = '/slfs1/users/zjz17/github/data/ptb_data/'
vocab_path = os.path.join(data_dir, 'ptb.vocab.txt')
train_path = os.path.join(data_dir, 'ptb.train.txt')
valid_path = os.path.join(data_dir, 'ptb.valid.txt')
test_path = os.path.join(data_dir, 'ptb.test.txt')

word2idx = read_dict(vocab_path)
ignore_label = word2idx.get('<pad>')

data_train, label_train = get_text_id(train_path, word2idx)
data_valid, label_valid = get_text_id(valid_path, word2idx)

# -----------------2. Params Defination ----------------------------------------
num_buckets = 1
batch_size = 32

#  network parameters
num_lstm_layer = 1

input_size = len(word2idx)
dropout = 0.0

num_embed = 512
num_hidden = 1024
num_label = len(word2idx)

init_h = [('ptb_l%d_init_h' % i, (batch_size, num_hidden)) for i in range(num_lstm_layer)]
init_c = [('ptb_l%d_init_c' % i, (batch_size, num_hidden)) for i in range(num_lstm_layer)]
init_states = init_h + init_c 
# training parameters
params_dir = 'params'
params_prefix = 'obama'
# program  parameters
seed = 1
np.random.seed(seed)


# ----------------------3. Data Iterator Defination ---------------------
train_iter = SequenceIter(
	data = data_train, 
	label = label_train, 
	pad = word2idx.get('<pad>'),
	init_states = init_states,
	batch_size = batch_size,
	num_buckets = num_buckets,
)
valid_iter = SequenceIter(
	data = data_valid, 
	label = label_valid, 
	pad = word2idx.get('<pad>'),
	init_states = init_states,
	batch_size = batch_size,
	num_buckets = num_buckets,
)

# ------------------4. Load paramters if exists ------------------------------

model_args = {}

if os.path.isfile('%s/%s-symbol.json' % (params_dir, params_prefix)):
	filelist = os.listdir(params_dir)
	paramfilelist = []
	for f in filelist:
		if f.startswith('%s-' % params_prefix) and f.endswith('.params'):
			paramfilelist.append( int(re.split(r'[-.]', f)[1]) )
	last_iteration = max(paramfilelist)
	print('laoding pretrained model %s/%s at epoch %d' % (params_dir, params_prefix, last_iteration))
	tmp = mx.model.FeedForward.load('%s/%s' % (params_dir, params_prefix), last_iteration)
	model_args.update({
		'arg_params' : tmp.arg_params,
		'aux_params' : tmp.aux_params,
		'begin_epoch' : tmp.begin_epoch
		})

# -----------------------5. Training ------------------------------------
def gen_sym(bucketkey):
	return rnn_unroll(
        num_layers = num_lstm_layer,
		seq_len = bucketkey,
		input_size = input_size,
		num_hidden = num_hidden,
		num_embed = num_embed,
		num_label = num_label,
		ignore_label = ignore_label,
        mode = 'lstm', 
        bi_directional = False,
        dropout = 0., 
        train = True
	)

model = mx.model.FeedForward(
	ctx = [mx.context.gpu(i) for i in range(1)],
	symbol = gen_sym,
	num_epoch = 10,
	optimizer = 'Adam',
	wd = 0.,
	initializer   = mx.init.Uniform(0.05),
	**model_args
)

if not os.path.exists(params_dir):
	os.makedirs(params_dir)

def perplexity(label, pred):
	label = label.T.reshape((-1,))
	loss = 0.
	num = 0.
	for i in range(pred.shape[0]):
		if int(label[i]) != ignore_label and int(label[i]) <= 10000:
			num += 1
			loss += -np.log(max(1e-10, pred[i][int(label[i])]))
	return np.exp(loss / num)

model.fit(
	X = train_iter,
	eval_data = valid_iter,
	eval_metric = mx.metric.np(perplexity),
	batch_end_callback = [mx.callback.Speedometer(batch_size, frequent = 100)],
	epoch_end_callback = [mx.callback.do_checkpoint('%s/%s' % (params_dir, params_prefix), 1)]
)
