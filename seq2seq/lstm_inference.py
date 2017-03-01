import mxnet as mx
import numpy as np

from collections import namedtuple
import math
import copy

from seq2seq import Seq2Seq

class BeamSearch(Seq2Seq):
    def __init__(self, num_lstm_layer, enc_data, enc_mask, enc_len, enc_input_size, 
        dec_input_size , num_hidden, num_embed, num_label, batch_size,
        arg_params, eos, unk, pad, 
        ctx = mx.cpu(), enc_dropout=0., dec_dropout = 0.0, output_dropout = 0.2):
        
        super(BeamSearch, self).__init__(
            enc_mode = 'lstm', 
            enc_num_layers = num_lstm_layer, 
            enc_len = enc_len,
            enc_input_size = enc_input_size, 
            enc_num_embed = num_embed, 
            enc_num_hidden = num_hidden,
            enc_dropout = enc_dropout, 
            enc_name = 'enc',
            enc_info_size = 28,
            dec_mode = 'lstm', 
            dec_num_layers = num_lstm_layer, 
            dec_len = 1,
            dec_input_size = dec_input_size, 
            dec_num_embed = num_embed, 
            dec_num_hidden = num_hidden,
            dec_num_label = num_label, 
            ignore_label = -1, 
            dec_dropout = dec_dropout, 
            dec_name = 'dec',
            output_dropout = output_dropout,
            train = False
        )

        self.eos = eos
        self.unk = unk
        self.pad = pad
        self.batch_size = batch_size

        self.encoder_symbol = self.encoder(
            mode = self.enc_mode, 
            num_layers = self.enc_num_layers, 
            enc_len = self.enc_len, 
            enc_input_size = self.enc_input_size, 
            num_embed = self.enc_num_embed, 
            num_hidden = self.enc_num_hidden, 
            enc_dropout = self.enc_dropout, 
            name = self.enc_name
        )
        init_c = [('%s_l%d_init_c' % (self.enc_name, l), (batch_size, num_hidden)) for l in range(num_lstm_layer)]
        init_h = [('%s_l%d_init_h' % (self.enc_name, l), (batch_size, num_hidden)) for l in range(num_lstm_layer)]
        enc_data_shape = [("enc_data", (batch_size, enc_len))]
        enc_mask_shape = [("enc_mask", (batch_size, enc_len))]
        enc_info_shape = [("enc_info", (batch_size, enc_len))]
        #enc_input_shapes = dict(init_c + init_h + enc_data_shape + enc_mask_shape + enc_info_shape)
        enc_input_shapes = dict(init_c + init_h + enc_data_shape + enc_mask_shape)
        # bind the network and provide the pretrained parameters
        self.encoder_executor = self.encoder_symbol.simple_bind(ctx = ctx, **enc_input_shapes)
        for key in self.encoder_executor.arg_dict.keys():
            if key in arg_params:
                arg_params[key].copyto(self.encoder_executor.arg_dict[key])
        # provide the input data and forward the network
        enc_data.copyto(self.encoder_executor.arg_dict["enc_data"])
        enc_mask.copyto(self.encoder_executor.arg_dict["enc_mask"])
        '''enc_info  = mx.nd.zeros((1, enc_len))
        for i in range(enc_len):
            enc_info[0,i] = sum(map(int, str(int(enc_data.asnumpy()[0,i])-3)))'''
        #enc_info.copyto(self.encoder_executor.arg_dict["enc_info"])
        
        self.encoder_executor.forward()

        # get the encoded vector for decoder hidden state initialization
        self.state_name = []
        for i in range(num_lstm_layer):
            self.state_name.append("%s_l%d_init_h" % (self.dec_name, i))
            self.state_name.append("%s_l%d_init_c" % (self.dec_name, i))
        self.init_states_dict = dict(zip(self.state_name, self.encoder_executor.outputs[0:]))

        # ---------------------------- 2. Define the decoder inference model ------------------
        self.decoder_symbol =  self.decoder(
            mode = self.dec_mode, 
            num_layers = self.dec_num_layers, 
            dec_len = self.dec_len, 
            dec_input_size = self.dec_input_size, 
            num_embed = self.dec_num_embed, 
            num_hidden = self.dec_num_hidden, 
            num_label = self.dec_num_label, 
            ignore_label = self.ignore_label, 
            dec_dropout = self.dec_dropout,
            output_dropout = self.output_dropout, 
            name = self.dec_name
        )
        
        init_c = [('%s_l%d_init_c' % (self.dec_name, l), (batch_size, num_hidden)) for l in range(num_lstm_layer)]
        init_h = [('%s_l%d_init_h' % (self.dec_name, l), (batch_size, num_hidden)) for l in range(num_lstm_layer)]
        dec_data_shape = [("dec_data", (batch_size,1))]
        dec_mask_shape = [("dec_mask", (batch_size,1))]
        dec_input_shapes = dict(init_c + init_h + dec_data_shape + dec_mask_shape)
        self.decoder_executor = self.decoder_symbol.simple_bind(ctx = ctx, **dec_input_shapes)
        for key in self.decoder_executor.arg_dict.keys():
            if key in arg_params:
                arg_params[key].copyto(self.decoder_executor.arg_dict[key])

        self.dec_data = mx.nd.zeros((batch_size,1))
        self.dec_mask = mx.nd.zeros((batch_size,1))

    def beam_search(self):
        self.beam = 10
        active_sentences = [(0,[self.eos], 0) ]
        ended_sentences = []
        states_list  = [self.init_states_dict]
        min_length = 0
        max_length = 20
        min_count = min(self.beam, len(active_sentences))

        for seqidx in xrange(max_length):
            tmp_sentences = []
            new_states_list = []
            for i in xrange(min_count):
                
                states_dict  = states_list[active_sentences[i][2]]
                for key in states_dict.keys():
                    states_dict[key].copyto(self.decoder_executor.arg_dict[key])

                dec_data = mx.nd.zeros((1, 1))
                dec_mask = mx.nd.ones((1, 1))
                tmp_data = np.zeros((1, 1))
                tmp_data[0] = active_sentences[i][1][-1]
                dec_data[:] = tmp_data
                dec_data.copyto(self.decoder_executor.arg_dict["dec_data"])
                dec_mask.copyto(self.decoder_executor.arg_dict["dec_mask"])
                
                self.decoder_executor.forward()

                new_states_dict = dict(zip(self.state_name, self.decoder_executor.outputs[1:]))
                tmp_states_dict = copy.deepcopy(new_states_dict)
                new_states_list.append(tmp_states_dict)

                prob = self.decoder_executor.outputs[0].asnumpy()
                
                # === this order is from small to big =====
                indecies = np.argsort(prob, axis = 1)[0]

                for j in xrange(self.beam):
                    score = active_sentences[i][0] + math.log(prob[0][indecies[-j-1]])
                    sent = active_sentences[i][1][:]
                    sent.extend([indecies[-j-1]])
                    if sent[-1] == self.eos:
                        if seqidx >= min_length:
                            ended_sentences.append((score, sent))
                    elif sent[-1] != self.unk and sent[-1] != self.pad:
                        tmp_sentences.append((score, sent, i))

            states_list = new_states_list[:]
            min_count = min(self.beam, len(tmp_sentences))
            active_sentences = sorted(tmp_sentences, reverse = True)[:min_count]

        result_sentences = []
        for sent in active_sentences:
            result_sentences.append((sent[0], sent[1]))
        for sent in ended_sentences:
            result_sentences.append(sent)
        result = min(self.beam, len(result_sentences), 10)
        result_sentences = sorted(result_sentences, reverse = True)[:result]
        return result_sentences

'''self.eos = eos
self.unk = unk
self.pad = pad
self.batch_size = batch_size

# -------------------------------1. get the encoded vector ------------------------------------------------
# provide the initial states and input shapes
self.encoder = encoder(num_lstm_layer, enc_len, enc_input_size,
                            num_embed, num_hidden, enc_dropout=enc_dropout)
init_c = [('l%d_init_c'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
init_h = [('l%d_init_h'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
enc_data_shape = [("enc_data", (batch_size,enc_len))]
enc_mask_shape = [("enc_mask", (batch_size,enc_len))]
enc_input_shapes = dict(init_c + init_h + enc_data_shape + enc_mask_shape)
# bind the network and provide the pretrained parameters
self.encoder_executor = self.encoder.simple_bind(ctx = ctx, **enc_input_shapes)
for key in self.encoder_executor.arg_dict.keys():
    if key in arg_params:
        arg_params[key].copyto(self.encoder_executor.arg_dict[key])
# provide the input data and forward the network
enc_data.copyto(self.encoder_executor.arg_dict["enc_data"])
enc_mask.copyto(self.encoder_executor.arg_dict["enc_mask"])
self.encoder_executor.forward()

# get the encoded vector for decoder hidden state initialization
self.state_name = []
for i in range(num_lstm_layer):
    self.state_name.append("l%d_init_c" % i)
    self.state_name.append("l%d_init_h" % i)
self.init_states_dict = dict(zip(self.state_name, self.encoder_executor.outputs[0:]))

# ---------------------------- 2. Define the decoder inference model ------------------
self.decoder = decoder_inference(num_lstm_layer, dec_input_size, 
                  num_hidden, num_embed, num_label, dec_dropout)

init_c = [('l%d_init_c'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
init_h = [('l%d_init_h'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
dec_data_shape = [("dec_data", (batch_size,))]
dec_mask_shape = [("dec_mask", (batch_size,))]
dec_input_shapes = dict(init_c + init_h + dec_data_shape + dec_mask_shape)
self.decoder_executor = self.decoder.simple_bind(ctx = ctx, **dec_input_shapes)
for key in self.decoder_executor.arg_dict.keys():
    if key in arg_params:
        arg_params[key].copyto(self.decoder_executor.arg_dict[key])

self.dec_data = mx.nd.zeros((batch_size,))
self.dec_mask = mx.nd.zeros((batch_size,))


def beam_search(self):
self.beam = 10
active_sentences = [(0,[self.eos], 0) ]
ended_sentences = []
states_list  = [self.init_states_dict]
min_length = 0
max_length = 20
min_count = min(self.beam, len(active_sentences))

for seqidx in xrange(max_length):
    tmp_sentences = []
    new_states_list = []
    for i in xrange(min_count):
        
        states_dict  = states_list[active_sentences[i][2]]
        for key in states_dict.keys():
            states_dict[key].copyto(self.decoder_executor.arg_dict[key])

        dec_data = mx.nd.zeros((1, ))
        dec_mask = mx.nd.ones((1, ))
        tmp_data = np.zeros((1, ))
        tmp_data[0] = active_sentences[i][1][-1]
        dec_data[:] = tmp_data
        dec_data.copyto(self.decoder_executor.arg_dict["dec_data"])
        dec_mask.copyto(self.decoder_executor.arg_dict["dec_mask"])
        
        self.decoder_executor.forward()

        new_states_dict = dict(zip(self.state_name, self.decoder_executor.outputs[1:]))
        tmp_states_dict = copy.deepcopy(new_states_dict)
        new_states_list.append(tmp_states_dict)

        prob = self.decoder_executor.outputs[0].asnumpy()
        
        # === this order is from small to big =====
        indecies = np.argsort(prob, axis = 1)[0]

        for j in xrange(self.beam):
            score = active_sentences[i][0] + math.log(prob[0][indecies[-j-1]])
            sent = active_sentences[i][1][:]
            sent.extend([indecies[-j-1]])
            if sent[-1] == self.eos:
                if seqidx >= min_length:
                    ended_sentences.append((score, sent))
            elif sent[-1] != self.unk and sent[-1] != self.pad:
                tmp_sentences.append((score, sent, i))

    states_list = new_states_list[:]
    min_count = min(self.beam, len(tmp_sentences))
    active_sentences = sorted(tmp_sentences, reverse = True)[:min_count]

result_sentences = []
for sent in active_sentences:
    result_sentences.append((sent[0], sent[1]))
for sent in ended_sentences:
    result_sentences.append(sent)
result = min(self.beam, len(result_sentences), 10)
result_sentences = sorted(result_sentences, reverse = True)[:result]
return result_sentences

def beam_search_batch(self):
self.beam = 10
# 1st is score, 2nd is the current sentence id, 3rd is the prev staten id  

active_sentences = [(np.zeros((self.batch_size, 1)), np.full((self.batch_size, 1), self.eos, dtype = float), 0)]

ended_sentences = []
states_list  = [self.init_states_dict]
min_length = 4
max_length = 18
min_count = min(self.beam, len(active_sentences))

for seqidx in xrange(max_length):
    tmp_sentences = []
    new_states_list = []
    for i in xrange(min_count):
        
        states_dict  = states_list[active_sentences[i][2]]
        for key in states_dict.keys():
            states_dict[key].copyto(self.decoder_executor.arg_dict[key])

        dec_data =  mx.nd.array(active_sentences[i][1][:,-1]).reshape((self.batch_size, ))
        dec_mask = mx.nd.ones((self.batch_size, ))

        dec_data.copyto(self.decoder_executor.arg_dict["dec_data"])
        dec_mask.copyto(self.decoder_executor.arg_dict["dec_mask"])
        
        self.decoder_executor.forward()

        new_states_dict = dict(zip(self.state_name, self.decoder_executor.outputs[1:]))
        tmp_states_dict = copy.deepcopy(new_states_dict)
        new_states_list.append(tmp_states_dict)

        prob = self.decoder_executor.outputs[0].asnumpy()

        # === this order is from small to big =====
        indecies = np.argsort(prob, axis = 1)

        for j in xrange(self.beam):
            score = active_sentences[i][0] + np.log(prob[ range(self.batch_size), indecies[:,-j-1] ]).reshape((self.batch_size, 1))
            sent = np.column_stack((active_sentences[i][1], indecies[:,-j-1]))
            tmp_sentences.append((score, sent, i))
            if sent[-1] == self.eos:
                if seqidx >= min_length:
                    ended_sentences.append((score, sent))
            elif sent[-1] != self.unk and sent[-1] != self.pad:
                tmp_sentences.append((score, sent, i))

    states_list = new_states_list[:]
    min_count = min(self.beam, len(tmp_sentences))
    active_sentences = sorted(tmp_sentences,key = lambda item: item[0].sum(), reverse = True)[:min_count]
result_sentences = []
for sent in active_sentences:
    result_sentences.append((sent[0], sent[1]))
for sent in ended_sentences:
    result_sentences.append(sent)
result = min(self.beam, len(result_sentences), 10)
result_sentences = sorted(result_sentences,key = lambda item: item[0].sum(), reverse = True)[:result]
return result_sentences'''