import mxnet as mx
import numpy as np

import re, os, sys, argparse, logging,collections
import codecs

from text_io import read_dict, get_enc_dec_text_id
from enc_dec_iter import EncoderDecoderIter, DummyIter
from seq2seq import Seq2Seq

logging.basicConfig(level = logging.DEBUG,
                    format = '%(asctime)s %(message)s', 
                    datefmt = '%m-%d %H:%M:%S %p',  
                    filename = 'Log',
                    filemode = 'w')
logger = logging.getLogger()
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
logger.addHandler(console)

DEBUG = True
# ----------------- 1. Process the data  ---------------------------------------
data_dir = '/slfs1/users/zjz17/github/data/sort'
vocab_file = 'q3.vocab'
train_file = 'q3.train'
valid_file = 'q3.valid'

enc_word2idx = read_dict(os.path.join(data_dir, vocab_file))
dec_word2idx = read_dict(os.path.join(data_dir, vocab_file))
ignore_label = enc_word2idx.get('<PAD>')

if DEBUG:
    print 'read_dict length:', len(enc_word2idx)

enc_data, dec_data,  = get_enc_dec_text_id(os.path.join(data_dir, train_file), enc_word2idx, dec_word2idx)
enc_valid, dec_valid, = get_enc_dec_text_id(os.path.join(data_dir, valid_file), enc_word2idx, dec_word2idx)

if DEBUG:
    print 'enc_data length: ' , len(enc_data), enc_data[0:1]
    print 'dec_data_length: ' , len(dec_data), dec_data[0:1]
    print 'enc_valid_length: ', len(enc_valid), enc_valid[0:1]
    print 'dec_valid_length: ', len(dec_valid), dec_valid[0:1]

# -----------------2. Params Defination ----------------------------------------
num_buckets = 1
batch_size = 16
frequent = len(enc_data) / batch_size / 10
#  network parameters
num_lstm_layer = 1

enc_input_size = len(enc_word2idx)
dec_input_size = len(dec_word2idx)
enc_dropout = 0.0
dec_dropout = 0.0
output_dropout = 0.2

num_embed = 100
num_hidden = 200
num_label = len(dec_word2idx)

init_h = [('enc_l%d_init_h' % i, (batch_size, num_hidden)) for i in range(num_lstm_layer)]
init_c = [('enc_l%d_init_c' % i, (batch_size, num_hidden)) for i in range(num_lstm_layer)]
init_states = init_h + init_c 
# training parameters
params_dir = 'params'
params_prefix = 'couplet'
# program  parameters
seed = 1
np.random.seed(seed)


# ----------------------3. Data Iterator Defination ---------------------
train_iter = EncoderDecoderIter(
                    enc_data = enc_data, 
                    dec_data = dec_data, 
                    pad = enc_word2idx.get('<PAD>'),
                    eos = enc_word2idx.get('<EOS>'),
                    init_states = init_states,
                    batch_size = batch_size,
                    num_buckets = num_buckets,
                    )
valid_iter = EncoderDecoderIter(
                    enc_data = enc_valid, 
                    dec_data = dec_valid, 
                    pad = enc_word2idx.get('<PAD>'),
                    eos = enc_word2idx.get('<EOS>'),
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
    seq2seq_model = Seq2Seq(
        enc_mode = 'lstm', 
        enc_num_layers = num_lstm_layer, 
        enc_len = bucketkey.enc_len,
        enc_input_size = enc_input_size, 
        enc_num_embed = num_embed, 
        enc_num_hidden = num_hidden,
        enc_dropout = enc_dropout, 
        enc_name = 'enc',
        enc_info_size = 28,
        dec_mode = 'lstm', 
        dec_num_layers = num_lstm_layer, 
        dec_len = bucketkey.dec_len,
        dec_input_size = dec_input_size, 
        dec_num_embed = num_embed, 
        dec_num_hidden = num_hidden,
        dec_num_label = num_label, 
        ignore_label = dec_word2idx.get('<PAD>'), 
        dec_dropout = dec_dropout, 
        dec_name = 'dec',
        output_dropout = output_dropout
    )
    return seq2seq_model.get_softmax()

def infer_shape(symbol = None, **shape):
    if symbol is None:
        return
    arg_name = symbol.list_arguments()
    out_name = symbol.list_outputs()
    print arg_name
    print out_name
    arg_shape, out_shape, aux_shape = symbol.infer_shape(**shape)
    print {'input': dict(zip(arg_name, arg_shape)),
    'output': dict(zip(out_name, out_shape))}
'''
input_shapes = {}
for term in train_iter.provide_data:
    input_shapes[term[0]] = term[1]
for term in train_iter.provide_label:
    input_shapes[term[0]] = term[1]
print input_shapes  
infer_shape(gen_sym(train_iter.default_bucket_key), 
    **input_shapes)

raw_input()
'''

class Customized(mx.init.Initializer):
    def __init__(self, enc_word2idx, dec_word2idx, scale):
        super(Customized, self).__init__()
        self.scale = scale
        self.enc_word2idx = enc_word2idx
        self.dec_word2idx = dec_word2idx
        self.word2vec = self.load_word2vec()
        self.enc_embed = self.get_embed_with_word2vec(self.enc_word2idx, self.word2vec)
        self.dec_embed = self.get_embed_with_word2vec(self.dec_word2idx, self.word2vec)

    def load_word2vec(self, file = 'stc_corpus.vector'):
        mydic = {}
        with codecs.open(file, encoding = 'utf-8') as f:
            f.readline()
            while True:
                str = f.readline()
                if str == '':
                    break;
                else:
                    lis = str.strip().split(' ')
                    tmp = [float(word) for word in lis[1:]]
                    mydic[lis[0]] = np.array(tmp)
        return mydic

    def get_embed_with_word2vec(self, word2idx, word2vec):
        embed_weight = []
        lis = sorted(word2idx.iteritems(), key=lambda d:d[1], reverse = False)
        total = len(word2idx)
        num = 0
        np.random.seed(1)
        for word, idx in lis:
            if word in word2vec:
                embed_weight.append(word2vec[word])
                num += 1
            else:
                vector = np.random.uniform(-0.05, 0.05, 400)
                embed_weight.append(vector)
        logging.info('total: %d, idx: %d' % (total, num))
        embed_weight = np.array(embed_weight)
        return embed_weight

    def _init_weight(self, name, arr):
        if name.endswith('enc_embed_weight'):
            enc_embed_weight = mx.nd.array(self.enc_embed)
            logging.info('enc_embed_weight with word2vec')
            enc_embed_weight.copyto(arr)
        elif name.endswith('dec_embed_weight'):
            dec_embed_weight = mx.nd.array(self.dec_embed)
            logging.info('enc_embed_weight with word2vec')
            dec_embed_weight.copyto(arr)
        else:
            mx.random.uniform(-self.scale, self.scale, out=arr)

model = mx.model.FeedForward(
    ctx = [mx.context.gpu(i) for i in range(1)],
    symbol = gen_sym,
    num_epoch = 70,
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
        if int(label[i]) != ignore_label:
            num += 1
            loss += -np.log(max(1e-10, pred[i][int(label[i])]))
    return np.exp(loss / num)

def customized_metric(label, pred):
    label = label.T.reshape((-1,))
    pred_indices = np.argmax(pred, axis = 1)
    num = 0
    for i in range(pred.shape[0]):
        if int(label[i]) != ignore_label:
            l = sum(map(int, str(int(label[i]-3)))) if label[i] >= 3 else 0
            p = sum(map(int, str(int(pred_indices[i]-3)))) if pred_indices[i] >= 3 else 0
            if l == p:
                num += 1
    return float(num) / pred.shape[0]

model.fit(
    X = train_iter,
    eval_data = valid_iter,
    eval_metric = mx.metric.np(customized_metric),
    batch_end_callback = [mx.callback.Speedometer(batch_size, frequent = frequent)],
    epoch_end_callback = [mx.callback.do_checkpoint('%s/%s' % (params_dir, params_prefix), 1)])
