#coding=utf-8

import sys
import copy
import math
import mxnet as mx
import numpy as np 
sys.path.append('..')
from rnn.rnn import GRU

class Seq2Seq(object):
    '''Sequence to sequence learning with neural networks
    The basic sequence to sequence learning network

    Note: you can't use gru as encoder and lstm as decoder
    because so makes the lstm cell has no initilization. 
    '''

    def __init__(self, enc_input_size, dec_input_size, enc_len, dec_len, num_label,
                share_embed_weight = False, is_train = True, ignore_label = 0):
        super(Seq2Seq, self).__init__()
        # ------------------- Parameter definition -------------------------
        self.enc_input_size = enc_input_size
        self.dec_input_size = dec_input_size
        self.enc_len = enc_len
        self.dec_len = dec_len
        self.num_label = num_label
        self.share_embed_weight = share_embed_weight
        self.is_train = is_train
        self.enc_num_embed = 512
        self.enc_num_hidden = 1024
        self.enc_name = 'enc'
        self.dec_num_embed = 512
        self.dec_num_hidden = 1024
        self.dec_name = 'dec'
        self.output_dropout = 0.2
        self.ignore_label = ignore_label

        if self.share_embed_weight:  # (for same language task, for example, dialog)
            self.embed_weight = mx.sym.Variable('embed_weight')
            self.enc_embed_weight = self.embed_weight
            self.dec_embed_weight = self.embed_weight
        else:  # (for multi languages task, for example, translation)
            self.enc_embed_weight = mx.sym.Variable('%s_embed_weight' % self.enc_name)
            self.dec_embed_weight = mx.sym.Variable('%s_embed_weight' % self.dec_name)

    def symbol_define(self):
        enc_data = mx.sym.Variable('%s_data' % self.enc_name)
        if self.is_train:
            enc_mask = mx.sym.Variable('%s_mask' % self.enc_name)
        else:
            enc_mask = None
        enc_embed = mx.sym.Embedding(
            data = enc_data, 
            input_dim = self.enc_input_size, 
            weight = self.enc_embed_weight, 
            output_dim = self.enc_num_embed, 
            name = '%s_embed' % self.enc_name
        )
        gru = GRU(num_hidden = self.enc_num_hidden, name = self.enc_name)
        enc_output, [enc_last_h] = gru.unroll(data = enc_embed, seq_len = self.enc_len, mask = enc_mask)

        dec_trans_h_temp = mx.sym.FullyConnected(
            data = enc_last_h, 
            num_hidden = self.dec_num_hidden,
            name = 'encode_to_decode_transform_weight'
        )
        dec_trans_h = mx.sym.Activation(dec_trans_h_temp, act_type = "tanh")
        dec_data = mx.sym.Variable('%s_data' % self.dec_name)
        if self.is_train:
            dec_mask = mx.sym.Variable('%s_mask' % self.dec_name)
        else:
            dec_mask = None
        dec_embed = mx.sym.Embedding(
            data = dec_data, 
            input_dim = self.dec_input_size, 
            weight = self.dec_embed_weight, 
            output_dim = self.dec_num_embed, 
            name = '%s_embed' % self.dec_name
        )
        gru = GRU(num_hidden = self.dec_num_hidden, name = self.dec_name)
        if not self.is_train:
            dec_init_h = mx.sym.Variable('%s_l0_init_h' % self.dec_name)
        else:
            dec_init_h = dec_trans_h
        dec_output, [dec_last_h] = gru.unroll(
            data = dec_embed, 
            seq_len = self.dec_len, 
            mask = dec_mask,
            begin_state = [dec_init_h]
        )
        hidden_concat = mx.sym.Concat(*dec_output, dim = 0)
        hidden_concat = mx.sym.Dropout(data=hidden_concat, p=self.output_dropout)
        pred = mx.sym.FullyConnected(
            data = hidden_concat, 
            num_hidden = self.num_label,
            name = '%s_pred' % self.dec_name
        )
        label = mx.sym.Variable('label')
        label = mx.sym.transpose(data = label)
        label = mx.sym.Reshape(data = label, shape = (-1, ))
        if self.is_train:
            sm = mx.sym.SoftmaxOutput(
                data = pred, 
                label = label, 
                name = 'softmax',
                use_ignore = True, 
                ignore_label = self.ignore_label
            )
            return sm
        else:
            sm = mx.sym.SoftmaxOutput(data = pred, name = 'softmax')      
            return dec_trans_h, mx.sym.Group([sm, dec_last_h])

    def predict(self, enc_data, arg_params, pad = 0, eos = 1, unk = 2):
        # ------------------------- bind data to symbol ---------------------------
        encoder, decoder = self.symbol_define()
        input_shapes = {}
        input_shapes['enc_data'] = (1, self.enc_len)
        encoder_executor = encoder.simple_bind(ctx = mx.cpu(), **input_shapes)
        for key in encoder_executor.arg_dict:
            if key in arg_params:
                arg_params[key].copyto(encoder_executor.arg_dict[key])
        enc_data.copyto(encoder_executor.arg_dict['enc_data'])
        encoder_executor.forward()
        dec_init_states = [('%s_l0_init_h' % self.dec_name, (1, self.dec_num_hidden))]
        state_name = [item[0] for item in dec_init_states]
        init_states_dict = dict(zip(state_name, encoder_executor.outputs[:]))
        dec_data_shape = [("dec_data", (1,1))]
        dec_input_shapes = dict(dec_data_shape + dec_init_states) 
        decoder_executor = decoder.simple_bind(ctx = mx.cpu(), **dec_input_shapes)
        for key in decoder_executor.arg_dict:
            if key in arg_params:
                arg_params[key].copyto(decoder_executor.arg_dict[key])
        
        # --------------------------- beam search ---------------------------------
        dec_data = mx.nd.zeros((1,1))
        beam = 10
        active_sentences = [(0,[eos], copy.deepcopy(init_states_dict))]
        ended_sentences = []
        min_length = 0
        max_length = 30
        min_count = min(beam, len(active_sentences))
        for seqidx in xrange(max_length):
            tmp_sentences = []
            for i in xrange(min_count):
                states_dict  = active_sentences[i][2]
                for key in states_dict.keys():
                    states_dict[key].copyto(decoder_executor.arg_dict[key])
                decoder_executor.arg_dict["dec_data"][:] = active_sentences[i][1][-1]
                decoder_executor.forward()
                new_states_dict = dict(zip(state_name, decoder_executor.outputs[1:]))
                tmp_states_dict = copy.deepcopy(new_states_dict)

                prob = decoder_executor.outputs[0].asnumpy()
                # === this order is from small to big =====
                indecies = np.argsort(prob, axis = 1)[0]

                for j in xrange(beam):
                    score = active_sentences[i][0] + math.log(prob[0][indecies[-j-1]])
                    sent = active_sentences[i][1][:]
                    sent.extend([indecies[-j-1]])
                    if sent[-1] == eos:
                        if seqidx >= min_length:
                            ended_sentences.append((score, sent))
                    elif sent[-1] != unk and sent[-1] != pad:
                        tmp_sentences.append((score, sent, tmp_states_dict))

            min_count = min(beam, len(tmp_sentences))
            active_sentences = sorted(tmp_sentences, reverse = True)[:min_count]
        result_sentences = []
        for sent in active_sentences:
            result_sentences.append((sent[0], sent[1]))
        for sent in ended_sentences:
            result_sentences.append(sent)
        result = min(beam, len(result_sentences), 10)
        result_sentences = sorted(result_sentences, reverse = True)[:result]
        return result_sentences