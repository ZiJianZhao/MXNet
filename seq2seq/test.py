import mxnet as mx
import numpy as np

import re, os, sys, argparse, logging,collections
import codecs

from text_io import read_dict, get_enc_dec_text_id
from lstm_inference import BeamSearch

DEBUG = True

data_dir = '/slfs1/users/zjz17/github/data/sort'
vocab_file = 'q3.vocab'
train_file = 'q3.train'
valid_file = 'q3.valid'

parser = argparse.ArgumentParser(description="Encoder-Decoder Model Inference")

parser.add_argument('--filename', default = os.path.join(data_dir, valid_file), 
    type = str, help='filename you wang to test')
parser.add_argument('--idx', default = 0, type = int,
    help='which line you want to test')
parser.add_argument('--mode', default = 'test', type = str,
    help='test one example or write examples results into a file')

args = parser.parse_args()
print args
filename = args.filename
idx = args.idx
mode = args.mode

enc_word2idx = read_dict(os.path.join(data_dir, vocab_file))
dec_word2idx = read_dict(os.path.join(data_dir, vocab_file))

print 'encoder dict length:', len(enc_word2idx)
print 'decoder dict length:', len(dec_word2idx)

enc_data, dec_data = get_enc_dec_text_id(filename, enc_word2idx, dec_word2idx)

print 'enc_data length: ' , len(enc_data), 
print 'example:', enc_data[0]
print 'dec_data_length: ' , len(dec_data)
print 'example:', dec_data[0]


# ------------------------------- Parameter Defination -------------------------------

#  network parameters
num_lstm_layer = 1
num_embed = 100
num_hidden = 200
num_label = len(dec_word2idx)

enc_input_size = len(enc_word2idx)
dec_input_size = len(dec_word2idx)
enc_dropout = 0.0
dec_dropout = 0.0
output_dropout = 0.2

# ---------------------------- Beam Search -----------------------------------------
dec_idx2word = {}
for k, v in dec_word2idx.items():
    dec_idx2word[v] = k

enc_idx2word = {}
for k, v in enc_word2idx.items():
    enc_idx2word[v] = k

_, arg_params, __ = mx.model.load_checkpoint('%s/%s' % ('params', 'couplet'), 16)

if mode == 'test':
    input_str = ""
    for i in enc_data[idx]:
        input_str += " " +  enc_idx2word[i]
    print "input sentence: ", input_str

    output_str = ""
    for i in dec_data[idx]:
        output_str += " " +  dec_idx2word[i]
    print "output sentence: ", output_str

    batch_size = 1
    enc_len = len(np.array(enc_data[idx]))
    enc_data  = mx.nd.array(np.array(enc_data[idx]).reshape(1, enc_len))
    enc_mask = mx.nd.array(np.ones((enc_len,)).reshape(1, enc_len))

    beam = BeamSearch(
    num_lstm_layer = num_lstm_layer, 
    enc_data = enc_data,
    enc_mask = enc_mask,
    enc_len = enc_len,
    enc_input_size = enc_input_size,
    dec_input_size = dec_input_size,
    num_hidden = num_hidden,
    num_embed = num_embed,
    num_label = num_label,
    batch_size = batch_size,
    arg_params = arg_params,
    eos = dec_word2idx.get('<EOS>'),
    unk = dec_word2idx.get('<UNK>'), 
    pad = dec_word2idx.get('<PAD>'),
    ctx=mx.cpu(), 
    enc_dropout=enc_dropout, 
    dec_dropout = dec_dropout,
    output_dropout = output_dropout)

    result_sentences = beam.beam_search()

    for pair in result_sentences:
        score = pair[0]
        sent = pair[1]
        mystr = ""
        for idx in sent:
            if dec_idx2word[idx] == '<EOS>':
                continue
            mystr += " " +  dec_idx2word[idx]
        print "%s\n" % mystr 
else:
    g = open('generate.txt', 'w')
    num = 0
    idx = -1
    while True:
        batch_size = 1
        idx += 1
        if num > 1000:
            break
        if idx > 0 and enc_data[idx] == enc_data[idx - 1]:
            continue
        num += 1
        input_str = ""
        for i in enc_data[idx]:
            input_str += " " +  enc_idx2word[i]
        print num, idx
        g.write('input:'+'\n'+ input_str.encode('utf8')+'\n')
        enc_len = len(np.array(enc_data[idx]))
        enc_data_t  = mx.nd.array(np.array(enc_data[idx]).reshape(1, enc_len))
        enc_mask = mx.nd.array(np.ones((enc_len,)).reshape(1, enc_len))
        beam = BeamSearch(
        num_lstm_layer = num_lstm_layer, 
        enc_data = enc_data_t,
        enc_mask = enc_mask,
        enc_len = enc_len,
        enc_input_size = enc_input_size,
        dec_input_size = dec_input_size,
        num_hidden = num_hidden,
        num_embed = num_embed,
        num_label = num_label,
        batch_size = batch_size,
        arg_params = arg_params,
        eos = dec_word2idx.get('<EOS>'),
        unk = dec_word2idx.get('<UNK>'), 
        pad = dec_word2idx.get('<PAD>'),
        ctx=mx.cpu(), 
        enc_dropout=enc_dropout, 
        dec_dropout = dec_dropout,
        output_dropout = output_dropout)
        result_sentences = beam.beam_search()
        g.write('outputs:'+'\n')
        for pair in result_sentences:
            score = pair[0]
            sent = pair[1]
            mystr = ""
            for ii in sent:
                if dec_idx2word[ii] == '<EOS>':
                    continue
                mystr += " " +  dec_idx2word[ii]
            g.write(mystr.encode('utf8')+'\n')
        g.write('==========================='+'\n')
    g.close()
