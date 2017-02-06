import mxnet as mx
import numpy as np

import re, os, sys, argparse, logging,collections
import codecs
from collections import namedtuple
from text_io import read_dict, get_enc_dec_text_id
from enc_dec_iter import EncoderDecoderIter, DummyIter
sys.path.append('..')
from seq2seq import Seq2Seq
Model = namedtuple("Model", ['executor', 'symbol'])

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

enc_word2idx = read_dict('../data/sort_test/vocab.txt')
dec_word2idx = read_dict('../data/sort_test/vocab.txt')
ignore_label = enc_word2idx.get('<PAD>')

if DEBUG:
    print 'read_dict length:', len(enc_word2idx)

enc_data, dec_data = get_enc_dec_text_id('../data/sort_test/tt.txt', enc_word2idx, dec_word2idx)
enc_valid, dec_valid = get_enc_dec_text_id('../data/sort_test/tt.txt', enc_word2idx, dec_word2idx)
if DEBUG:
    print 'enc_data length: ' , len(enc_data), enc_data[0:1]
    print 'dec_data_length: ' , len(dec_data), dec_data[0:1]
    print 'enc_valid_length: ', len(enc_valid), enc_valid[0:1]
    print 'dec_valid_length: ', len(dec_valid), dec_valid[0:1]

# -----------------2. Params Defination ----------------------------------------
num_buckets = 1
batch_size = 1

#  network parameters
num_lstm_layer = 1

enc_input_size = len(enc_word2idx)
dec_input_size = len(dec_word2idx)
enc_dropout = 0.0
dec_dropout = 0.0
output_dropout = 0.2

num_embed = 512
num_hidden = 1024
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

def setup_model(
    gen_sym, data_batch, ctx = mx.gpu(0)
):
    # define symbol

    bucketkey = data_batch.bucket_key
    symbol = gen_sym(bucketkey)
    print symbol.list_arguments()
    # bind symbol to executor
    input_shapes = data_batch.provide_data + data_batch.provide_label
    dic_shapes = {}
    for item in input_shapes:
        dic_shapes[item[0]] = item[1]
    arg_shape, out_shape, aux_shape = symbol.infer_shape(**dic_shapes)
    arg_arrays = [mx.nd.zeros(s, ctx) for s in arg_shape]
    args_grad = {}
    for shape, name in zip(arg_shape, symbol.list_arguments()):
        if name in dic_shapes.keys(): # input, output
            continue
        args_grad[name] = mx.nd.zeros(shape, mx.gpu(0))

    executor = symbol.bind(
        ctx = mx.gpu(0), 
        args = arg_arrays, 
        args_grad = args_grad, 
        grad_req = 'write'
    )
    
    # initialization
    arg_arrays = dict(zip(symbol.list_arguments(), executor.arg_arrays))
    '''for name, arr in arg_arrays.items():
        if name not in input_shapes:
            initializer(name, arr)'''
    for name, arr in arg_arrays.items():
        if name not in dic_shapes:
            if re.match('.*bias', name):
                bias = mx.nd.array(np.zeros(arr.shape))
                bias.copyto(arr)
            else:
                mx.init.Uniform(0.05)(name, arr)

    return Model(
        executor = executor, 
        symbol = symbol
    )

def perplexity(label, pred):
    label = label.T.reshape((-1,))
    loss = 0.
    num = 0.
    for i in range(pred.shape[0]):
        if int(label[i]) != ignore_label:
            num += 1
            loss += -np.log(max(1e-10, pred[i][int(label[i])]))
    '''print pred[i][:]
    print '(', num, loss, ')', 
    print '\n'
    '''
    return np.exp(loss / num)


def train_model(
    model, train_iter, valid_iter, batch_size, 
    optimizer = 'adam', max_grad_norm = 5.0,
    learning_rate = 0.0005, epochs = 200
):
    # optimizer definition
    opt = mx.optimizer.create(optimizer)
    opt.wd = 0.000
    updater = mx.optimizer.get_updater(opt)

    # metric definition
    train_metric = mx.metric.np(perplexity)
    valid_metric = mx.metric.np(perplexity)
    input_shapes = data_batch.provide_data + data_batch.provide_label
    dic_shapes = {}
    for item in input_shapes:
        dic_shapes[item[0]] = item[1]
    # training
    executor = model.executor
    symbol = model.symbol
    arg_arrays = dict(zip(symbol.list_arguments(), executor.arg_arrays))
    for epoch in range(epochs):
        print '-------------------begin--------------------------'
        '''for name, arr in arg_arrays.items():
            if name in dic_shapes:
                continue
            else:
                print name, np.linalg.norm(arr.asnumpy())'''
        print '---------------------------------------------------'
        train_iter.reset()
        valid_iter.reset()
        train_metric.reset()
        valid_metric.reset()

        t = 0
        for batch in train_iter:
            # Copy data to executor input. Note the [:].
            
            par = {}
            for i in range(len(batch.data_names)):
                par[batch.data_names[i]] = batch.data[i]
            for i in range(len(batch.label_names)):
                par[batch.label_names[i]] = batch.label[i]        
            for name, arr in arg_arrays.items():
                if name in dic_shapes:
                    par[name].copyto(arr)
                #print name, np.linalg.norm(arr.asnumpy()), arr.asnumpy().reshape((-1,))[0]
            
            # Forward
            executor.forward(is_train=True)
            
            '''print executor.outputs[2].asnumpy().shape
            print executor.outputs[3].asnumpy().shape
            raw_input()'''
            # Backward
            executor.backward()

            # Update
            
            for i, pair in enumerate(zip(symbol.list_arguments(), executor.arg_arrays, executor.grad_arrays)):
                name, weight, grad = pair
                if name in dic_shapes:
                    continue
                updater(i, grad, weight)

            # metric update
            train_metric.update(batch.label, executor.outputs)
            pred =  executor.outputs[0].asnumpy()
            label = executor.arg_dict['label'].asnumpy()
            print label
            print '-------------------train-------------------------'
            print perplexity(label, pred)
            print '-------------------------------------------------'
            print 'epoch: %d, iter: %d, perplexity: %.3f' % (epoch, t, float(train_metric.get()[1]))
        
        for batch in valid_iter:
            par = {}
            for i in range(len(batch.data_names)):
                par[batch.data_names[i]] = batch.data[i]
            for i in range(len(batch.label_names)):
                par[batch.label_names[i]] = batch.label[i]        
            for name, arr in arg_arrays.items():
                if name in dic_shapes:
                    par[name].copyto(arr)
                #print name, np.linalg.norm(arr.asnumpy()), arr.asnumpy().reshape((-1,))[0]
            executor.forward(is_train = False)
            
            pred =  executor.outputs[0].asnumpy()
            label = executor.arg_dict['label'].asnumpy()
            print label
            print '---------------------valid------------------------'
            print perplexity(label, pred)
            print '---------------------------------------------------'
            valid_metric.update(batch.label, executor.outputs)
        curr_acc = valid_metric.get()[1]
        print '==================================='
        print 'epoch: %d, validation accuracy: %.5f' % (epoch, valid_metric.get()[1])
        print '-------------------end--------------------------'
        '''for name, arr in arg_arrays.items():
            if name in dic_shapes:
                continue
            else:
                print name, np.linalg.norm(arr.asnumpy())'''
        print '---------------------------------------------------'

for data_batch in train_iter:
    model = setup_model(gen_sym, data_batch)
    break

train_model(
    model, train_iter, valid_iter, batch_size, 
    optimizer = 'adam', max_grad_norm = 5.0,
    learning_rate = 0.0005, epochs = 200
)