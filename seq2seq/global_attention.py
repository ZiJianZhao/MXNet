#coding=utf-8

import sys, os
import copy
import math
import mxnet as mx
import numpy as np 
sys.path.append('..')
from rnn.rnn import GRU
from enc_dec_iter import EncoderDecoderIter, read_dict, get_enc_dec_text_id
from eval_and_visual import draw_confusion_matrix

class GlobalSeq2Seq(object):
    '''Sequence to sequence learning with neural networks
    The basic sequence to sequence learning network

    Note: you can't use gru as encoder and lstm as decoder
    because so makes the lstm cell has no initilization. 
    '''

    def __init__(self, enc_input_size, dec_input_size, enc_len, dec_len, num_label,
                share_embed_weight = False, is_train = True, ignore_label = 0):
        super(GlobalSeq2Seq, self).__init__()
        # ------------------- Parameter definition -------------------------
        ''' The layer is 1, if you want to change it, there are some things to correct '''
        self.enc_input_size = enc_input_size
        self.dec_input_size = dec_input_size
        self.enc_len = enc_len
        self.dec_len = dec_len
        self.num_label = num_label
        self.share_embed_weight = share_embed_weight
        self.is_train = is_train
        self.enc_num_embed = 400 #512
        self.enc_num_hidden = 800 #1024
        self.enc_name = 'enc'
        self.bidirectional = False
        self.dec_num_embed = 400 #512
        self.dec_num_hidden = 800 #1024
        self.dec_name = 'dec'
        self.output_dropout = 0.
        self.ignore_label = ignore_label

        if self.share_embed_weight:  # (for same language task, for example, dialog)
            self.embed_weight = mx.sym.Variable('embed_weight')
            self.enc_embed_weight = self.embed_weight
            self.dec_embed_weight = self.embed_weight
        else:  # (for multi languages task, for example, translation)
            self.enc_embed_weight = mx.sym.Variable('%s_embed_weight' % self.enc_name)
            self.dec_embed_weight = mx.sym.Variable('%s_embed_weight' % self.dec_name)

    def encoder(self):
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
        # encoder 
        forward_gru = GRU(num_hidden = self.enc_num_hidden, name = '%s_forward' % self.enc_name)
        forward_enc_output, [forward_enc_last_h] = forward_gru.unroll(
            data = enc_embed, 
            seq_len = self.enc_len, 
            mask = enc_mask, 
            forward = True,
            merge_outputs = True
        )
        backwrd_gru = GRU(num_hidden = self.enc_num_hidden, name = '%s_backward' % self.enc_name)
        backward_enc_output, [backward_enc_last_h] = backwrd_gru.unroll(
            data = enc_embed, 
            seq_len = self.enc_len, 
            mask = enc_mask, 
            forward = False,
            merge_outputs = True
        )
        enc_output = mx.sym.Concat(*[forward_enc_output, backward_enc_output], dim = 2)
        if self.bidirectional:     
            return enc_mask, enc_output, forward_enc_last_h
        else:
            return enc_mask, forward_enc_output, forward_enc_last_h


    class DotAttention(object):
        '''Effective Approaches to Attention-based Neural Machine Translation
            e = enc_hiddens^T * dec_hidden

            This class need that the enc_hiddens num_features equals to the dec_hidden num_features 
        '''
        def __init__(self, is_train):
            super(GlobalSeq2Seq.DotAttention, self).__init__()
            self.is_train = is_train

        def __call__(self, enc_hiddens, dec_hidden, enc_len, enc_mask):
            target_hidden = mx.sym.Reshape(dec_hidden, shape = (-2, 1))
            source_target_atten = mx.sym.batch_dot(enc_hiddens, target_hidden)
            # if want to add mask, add mask here
            attention = mx.sym.SoftmaxActivation(data = source_target_atten, mode = 'channel')
            # if want to add mask, add mask here
            if self.is_train:
                temp_mask = mx.sym.Reshape(enc_mask, shape=(-2, 1))
                attention = mx.sym.broadcast_mul(attention, temp_mask)
                attention_sum = mx.sym.sum(attention, axis=1, keepdims=True)
                attention_sum = mx.sym.broadcast_axis(attention_sum, axis=1, size=enc_len)
                attention = mx.sym.broadcast_div(attention, attention_sum)
            return attention

    class ConcatAttention(object):
        '''Effective Approaches to Attention-based Neural Machine Translation
            e = W[enc_hiddens; dec_hidden]
        '''
        def __init__(self, enc_num_hidden, dec_num_hidden, is_train):
            super(GlobalSeq2Seq.ConcatAttention, self).__init__()
            self.is_train = is_train
            self.enc_num_hidden = enc_num_hidden
            self.dec_num_hidden = dec_num_hidden
            self.weight = mx.sym.Variable('source_target_concat_weight', shape = (2 * self.enc_num_hidden + self.dec_num_hidden, 1))       

        def __call__(self, enc_hiddens, dec_hidden, enc_len, enc_mask):
            target_hidden = mx.sym.expand_dims(dec_hidden, axis = 1)
            target_hidden = mx.sym.broadcast_axis(target_hidden, axis = 1, size = enc_len)
            source_target_hidden = mx.sym.Concat(*[enc_hiddens, target_hidden], dim = 2)
            source_target_atten = mx.sym.dot(source_target_hidden, self.weight)
            attention = mx.sym.SoftmaxActivation(data = source_target_atten, mode = 'channel')
            # if want to add mask, add mask here
            if self.is_train:
                temp_mask = mx.sym.Reshape(enc_mask, shape=(-2, 1))
                attention = mx.sym.broadcast_mul(attention, temp_mask)
                attention_sum = mx.sym.sum(attention, axis=1, keepdims=True)
                attention_sum = mx.sym.broadcast_axis(attention_sum, axis=1, size=enc_len)
                attention = mx.sym.broadcast_div(attention, attention_sum)
            return attention
    
    class GeneralAttention(object):
        '''Effective Approaches to Attention-based Neural Machine Translation
            e = enc_hiddens * W * dec_hidden
        '''
        def __init__(self, enc_num_hidden, dec_num_hidden, is_train):
            super(GlobalSeq2Seq.GeneralAttention, self).__init__()
            self.is_train = is_train
            self.enc_num_hidden = enc_num_hidden
            self.dec_num_hidden = dec_num_hidden
            self.weight = mx.sym.Variable('source_target_multi_weight', shape = (2 * self.enc_num_hidden, self.dec_num_hidden))       

        def __call__(self, enc_hiddens, dec_hidden, enc_len, enc_mask):
            hidden = mx.sym.dot(enc_hiddens, self.weight)
            target_hidden = mx.sym.Reshape(dec_hidden, shape = (-2, 1))
            source_target_atten = mx.sym.batch_dot(hidden, target_hidden)
            attention = mx.sym.SoftmaxActivation(data = source_target_atten, mode = 'channel')
            # if want to add mask, add mask here
            if self.is_train:
                temp_mask = mx.sym.Reshape(enc_mask, shape=(-2, 1))
                attention = mx.sym.broadcast_mul(attention, temp_mask)
                attention_sum = mx.sym.sum(attention, axis=1, keepdims=True)
                attention_sum = mx.sym.broadcast_axis(attention_sum, axis=1, size=enc_len)
                attention = mx.sym.broadcast_div(attention, attention_sum)
            return attention

    class NolinearAttention(object):
        '''Neural Machine Translation By Jointly Learning to Align and Translate
            e = v^T * tanh(w * enc_hiddens + u * dec_hidden)
        '''
        def __init__(self, enc_num_hidden, dec_num_hidden, is_train, atten_dim = 512):
            super(GlobalSeq2Seq.NolinearAttention, self).__init__()
            self.is_train = is_train
            self.enc_num_hidden = enc_num_hidden
            self.dec_num_hidden = dec_num_hidden
            self.atten_dim = atten_dim
            self.w_weight = mx.sym.Variable('source_attenion_weight', shape = (self.enc_num_hidden , self.atten_dim))
            self.u_weight = mx.sym.Variable('target_attenion_weight', shape = (self.dec_num_hidden, self.atten_dim))
            self.v_weight = mx.sym.Variable('attention_weight', shape = (self.atten_dim, 1))            

        def __call__(self, enc_hiddens, dec_hidden, enc_len, enc_mask):
            target_hidden = mx.sym.dot(dec_hidden, self.u_weight)
            target_hidden = mx.sym.expand_dims(target_hidden, axis=1)
            target_hidden = mx.sym.broadcast_axis(target_hidden, axis = 1, size = enc_len)
            source_hidden = mx.sym.dot(enc_hiddens, self.w_weight)
            temp = mx.sym.broadcast_add(target_hidden, source_hidden, name = 'error')
            source_target_hidden = mx.sym.Activation(temp, act_type="tanh")
            source_target_atten = mx.sym.dot(source_target_hidden, self.v_weight)
            attention = mx.sym.SoftmaxActivation(data = source_target_atten, mode = 'channel')
            # if want to add mask, add mask here
            if self.is_train:
                temp_mask = mx.sym.Reshape(enc_mask, shape=(-2, 1))
                attention = mx.sym.broadcast_mul(attention, temp_mask)
                attention_sum = mx.sym.sum(attention, axis=1, keepdims=True)
                attention_sum = mx.sym.broadcast_axis(attention_sum, axis=1, size=enc_len)
                attention = mx.sym.broadcast_div(attention, attention_sum)
            return attention

    def get_attention(self):
        return self.symbol_define(attention_type = 'nolinear', get_attention = True)

    def get_sofxmax(self):
        return self.symbol_define(attention_type = 'nolinear', get_attention = False)


    def symbol_define(self, attention_type = 'nolinear', get_attention = False):
        '''
            Inputs:
                attention_type: dot, concat, general, nolinear
                get_attention: get the variable for attention graph 
        '''
        enc_mask, enc_output, forward_enc_last_h = self.encoder()

        # Transform encoder last hidden state to decoder init state
        dec_trans_h_temp = mx.sym.FullyConnected(
            data = forward_enc_last_h, 
            num_hidden = self.dec_num_hidden,
            name = 'encode_to_decode_transform_weight'
        )
        dec_trans_h = mx.sym.Activation(dec_trans_h_temp, act_type = "tanh")

        # decoder input processing        
        dec_data = mx.sym.Variable('%s_data' % self.dec_name)
        if self.is_train:
            dec_mask = mx.sym.Variable('%s_mask' % self.dec_name)
            mask = mx.sym.SliceChannel(
                data = dec_mask,
                num_outputs = self.dec_len,
                axis = 1,
                squeeze_axis = True,
                name = '%s_mask_slice_channel' % self.dec_name
            )
        else:
            dec_mask = None
        dec_embed = mx.sym.Embedding(
            data = dec_data, 
            input_dim = self.dec_input_size, 
            weight = self.dec_embed_weight, 
            output_dim = self.dec_num_embed, 
            name = '%s_embed' % self.dec_name
        )
        embed = mx.sym.SliceChannel(
            data = dec_embed,
            num_outputs = self.dec_len,
            axis = 1,
            squeeze_axis = True,
            name = '%s_embed_slice_channel' % self.dec_name
        )
        if not self.is_train:
            dec_init_h = mx.sym.Variable('%s_l0_init_h' % self.dec_name) 
            enc_hidden = mx.sym.Variable('enc_hidden')
        else:
            dec_init_h = dec_trans_h
            enc_hidden = enc_output
        
        # Attention function selection
        if attention_type == 'nolinear':
            attention_func = GlobalSeq2Seq.NolinearAttention(self.enc_num_hidden, self.dec_num_hidden, self.is_train)
        elif attention_type == 'concat':
            attention_func = GlobalSeq2Seq.ConcatAttention(self.enc_num_hidden, self.dec_num_hidden, self.is_train)
        elif attention_type == 'dot':
            attention_func = GlobalSeq2Seq.DotAttention(self.is_train)
        elif attention_type == 'general':
            attention_func = GlobalSeq2Seq.GeneralAttention(self.enc_num_hidden, self.dec_num_hidden, self.is_train)
        else:
            raise NameError, 'Attention types: nolinear, concat, dot, general'

        # decoder
        dec_output = []
        attentions_list = [] 
        states = [dec_init_h]
        gru = GRU(num_hidden = self.dec_num_hidden, name = self.dec_name)
        for i in range(self.dec_len):
            attention =attention_func(enc_hidden, states[0], self.enc_len, enc_mask)
            attentions_list.append(attention)
            context_vector_pre = mx.sym.broadcast_mul(enc_hidden, attention)
            context_vector = mx.sym.sum(context_vector_pre, axis = 1)
            data = mx.sym.Concat(*[embed[i], context_vector], dim= 1)
            if self.is_train:
                output, states = gru(data, states, mask[i])
            else:
                output, states = gru(data, states)
            dec_output.append(output)

        # here for attention weight graph
        if get_attention:
            atten = mx.sym.Concat(*attentions_list, dim= 0)
            return atten

        # softmax 
        dec_last_h = states[0]
        dec_output = [mx.symbol.expand_dims(i, axis=1) for i in dec_output]
        dec_output = mx.symbol.Concat(*dec_output, dim=1)
        hidden_concat = mx.sym.Reshape(dec_output, shape=(-1, self.dec_num_hidden))
        #hidden_concat = mx.sym.Dropout(data = hidden_concat, p = self.output_dropout)
        pred = mx.sym.FullyConnected(
            data = hidden_concat, 
            num_hidden = self.num_label,
            name = '%s_pred' % self.dec_name
        )
        label = mx.sym.Variable('label')
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
            return mx.sym.Group([dec_trans_h, enc_output]), mx.sym.Group([sm, dec_last_h])


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
        enc_hidden = encoder_executor.outputs[1]
        dec_init_states = [('%s_l0_init_h' % self.dec_name, (1, self.dec_num_hidden))]
        state_name = [item[0] for item in dec_init_states]
        init_states_dict = dict(zip(state_name, [encoder_executor.outputs[0]]))
        dec_data_shape = [("dec_data", (1,1))]
        enc_hidden_shape = [('enc_hidden', (1, self.enc_len, self.enc_num_hidden * 2))]
        dec_input_shapes = dict(dec_data_shape + dec_init_states + enc_hidden_shape)
        decoder_executor = decoder.simple_bind(ctx = mx.cpu(), **dec_input_shapes)
        for key in decoder_executor.arg_dict:
            if key in arg_params:
                arg_params[key].copyto(decoder_executor.arg_dict[key])
        # --------------------------- beam search ---------------------------------
        beam = 10
        active_sentences = [(0,[eos], copy.deepcopy(init_states_dict))]
        ended_sentences = []
        min_length = 1
        max_length = 30
        min_count = min(beam, len(active_sentences))
        for seqidx in xrange(max_length):
            tmp_sentences = []
            for i in xrange(min_count):
                states_dict  = active_sentences[i][2]
                for key in states_dict.keys():
                    states_dict[key].copyto(decoder_executor.arg_dict[key])
                decoder_executor.arg_dict["dec_data"][:] = active_sentences[i][1][-1]
                decoder_executor.arg_dict['enc_hidden'][:] = enc_hidden.asnumpy()
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
        #result = min(beam, len(result_sentences), 10)
        #result_sentences = sorted(result_sentences, reverse = True)[:result]
        result_sentences = sorted(result_sentences, reverse = True)        
        return result_sentences

    def couplet_predict(self, enc_data, arg_params, pad = 0, eos = 1, unk = 2):
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
        enc_hidden = encoder_executor.outputs[1]
        dec_init_states = [('%s_l0_init_h' % self.dec_name, (1, self.dec_num_hidden))]
        state_name = [item[0] for item in dec_init_states]
        init_states_dict = dict(zip(state_name, [encoder_executor.outputs[0]]))
        dec_data_shape = [("dec_data", (1,1))]
        enc_hidden_shape = [('enc_hidden', (1, self.enc_len, self.enc_num_hidden * 2))]
        dec_input_shapes = dict(dec_data_shape + dec_init_states + enc_hidden_shape)
        decoder_executor = decoder.simple_bind(ctx = mx.cpu(), **dec_input_shapes)
        for key in decoder_executor.arg_dict:
            if key in arg_params:
                arg_params[key].copyto(decoder_executor.arg_dict[key])
        # --------------------------- beam search ---------------------------------
        beam = 20
        active_sentences = [(0,[eos], copy.deepcopy(init_states_dict))]
        ended_sentences = []
        min_length = 1
        max_length = enc_data.shape[1]
        min_count = min(beam, len(active_sentences))
        for seqidx in xrange(max_length):
            tmp_sentences = []
            for i in xrange(min_count):
                states_dict  = active_sentences[i][2]
                for key in states_dict.keys():
                    states_dict[key].copyto(decoder_executor.arg_dict[key])
                decoder_executor.arg_dict["dec_data"][:] = active_sentences[i][1][-1]
                decoder_executor.arg_dict['enc_hidden'][:] = enc_hidden.asnumpy()
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
                    if sent[-1] != eos and sent[-1] != unk and sent[-1] != pad:
                        tmp_sentences.append((score, sent, tmp_states_dict))

            min_count = min(beam, len(tmp_sentences))
            active_sentences = sorted(tmp_sentences, reverse = True)[:min_count]
        result_sentences = []
        for sent in active_sentences:
            result_sentences.append((sent[0], sent[1][1:]))
        #result = min(beam, len(result_sentences), 10)
        #result_sentences = sorted(result_sentences, reverse = True)[:result]
        result_sentences = sorted(result_sentences, reverse = True)        
        return result_sentences

def get_attentions(source, target, epoch):
    task = 'couplet'
    task_dir = '/slfs1/users/zjz17/github/data/couplet/'
    data_dir = task_dir + 'data/'
    params_dir = task_dir + 'global' + '_params/'
    params_prefix = 'couplet'
    share_embed_weight = True
    if share_embed_weight:
        enc_vocab_file = 'alllist.txt'
        dec_vocab_file = 'alllist.txt'
    else:
        enc_vocab_file = 'shanglist.txt'
        dec_vocab_file = 'xialist.txt'
    train_file = 'train.txt'
    valid_file = 'valid.txt'
    test_file = 'test.txt'
    enc_word2idx = read_dict(os.path.join(data_dir, enc_vocab_file))
    dec_word2idx = read_dict(os.path.join(data_dir, dec_vocab_file))

    source_list = source.strip().split()
    target_list = target.strip().split()
    source_data = [enc_word2idx.get(s) if enc_word2idx.get(s) is not None else enc_word2idx.get('<unk>') for s in source_list]
    target_data = [dec_word2idx.get(s) if dec_word2idx.get(s) is not None else dec_word2idx.get('<unk>') for s in target_list]
    enc_len = len(source_data)
    dec_len = len(target_data)
    enc_data = mx.nd.array(np.array(source_data)).reshape((1, enc_len))
    dec_data = mx.nd.array(np.array(target_data)).reshape((1, dec_len))
    batch_size = 1 
    enc_mask = mx.nd.ones((1, enc_len))
    dec_mask = mx.nd.ones((1, dec_len))

    data_all = [enc_data, enc_mask, dec_data, dec_mask]
    label_all = []
    data_names = ['enc_data', 'enc_mask', 'dec_data', 'dec_mask']
    label_names = []

    data_batch = mx.io.DataBatch(
        data = data_all,
        label = [],
        provide_data = zip(data_names, [data.shape for data in data_all]),
        provide_label = []
    )

    seq2seq = GlobalSeq2Seq(
        enc_input_size = len(enc_word2idx), 
        dec_input_size = len(dec_word2idx),
        enc_len = enc_len,
        dec_len = dec_len,
        num_label = len(dec_word2idx),
        share_embed_weight = share_embed_weight,
        is_train = True
    ).symbol_define(get_attention = True)
    mod = mx.mod.Module(seq2seq, 
        data_names = ['enc_data', 'enc_mask', 'dec_data', 'dec_mask'], 
        label_names = [],
        context = [mx.cpu()])
    sym, arg_params, aux_params = mx.model.load_checkpoint('%s%s' % (params_dir, params_prefix), epoch)
    provide_data = [('enc_data' , (batch_size, enc_len)),
                    ('enc_mask' , (batch_size, enc_len)),
                  ('dec_data' , (batch_size, dec_len)),
                  ('dec_mask' , (batch_size, dec_len))]
    mod.bind(data_shapes=provide_data)
    mod.set_params(arg_params=arg_params, aux_params=aux_params)
    mod.forward(data_batch) 
    outputs = mod.get_outputs()
    
    attention = outputs[0].asnumpy().reshape(dec_len, enc_len)
    return attention

def calculate_attention_metric(epoch):
    task_dir = '/slfs1/users/zjz17/github/data/couplet/'
    data_dir = task_dir + 'data/'
    params_dir = task_dir + 'global' + '_params/'
    params_prefix = 'couplet'
    share_embed_weight = True
    if share_embed_weight:
        enc_vocab_file = 'alllist.txt'
        dec_vocab_file = 'alllist.txt'
    else:
        enc_vocab_file = 'shanglist.txt'
        dec_vocab_file = 'xialist.txt'
    train_file = 'train.txt'
    valid_file = 'valid.txt'
    test_file = 'test.txt'
    enc_word2idx = read_dict(os.path.join(data_dir, enc_vocab_file))
    dec_word2idx = read_dict(os.path.join(data_dir, dec_vocab_file))

    enc_test, dec_test = get_enc_dec_text_id(os.path.join(data_dir, test_file), enc_word2idx, dec_word2idx)
    num = 10
    enc_test = enc_test[:num]
    dec_test = dec_test[:num]
    test_iter = EncoderDecoderIter(
        enc_data = enc_test, 
        dec_data = dec_test, 
        batch_size = 1, 
        num_buckets = 1, 
        pad = enc_word2idx.get('<pad>'), 
        eos = enc_word2idx.get('<eos>')
    )
    sym, arg_params, aux_params = mx.model.load_checkpoint('%s%s' % (params_dir, params_prefix), epoch)

    metric = 0
    for data_batch in test_iter:
        length = data_batch.bucket_key.enc_len
        seq2seq = GlobalSeq2Seq(
            enc_input_size = len(enc_word2idx), 
            dec_input_size = len(dec_word2idx),
            enc_len = data_batch.bucket_key.enc_len,
            dec_len = data_batch.bucket_key.dec_len,
            num_label = len(dec_word2idx),
            share_embed_weight = share_embed_weight,
            is_train = True
        ).symbol_define(get_attention = True)
        mod = mx.mod.Module(seq2seq, 
            data_names = ['enc_data', 'enc_mask', 'dec_data', 'dec_mask'], 
            label_names = [],
            context = [mx.cpu()]
        )

        mod.bind(data_shapes=test_iter.provide_data)
        mod.set_params(arg_params=arg_params, aux_params=aux_params)
        mod.forward(data_batch) 
        outputs = mod.get_outputs()
        attention = outputs[0].asnumpy().reshape(data_batch.bucket_key.enc_len, data_batch.bucket_key.dec_len)
        diag = np.eye(length)
        metric += np.fabs(attention - diag)
    print metric
        

if __name__ == '__main__':
    #source = u'十 月 塞 边 ， 飒 飒 寒 霜 惊 戍 旅'
    # attention here, the first input word in decoder is eos to predict the correspoding true first word 
    #target_output = u'三 东 江 上 ， 漫 漫 朔 雪 冷 渔 翁 <eos>'
    #target_input = u'<eos> 三 东 江 上 ， 漫 漫 朔 雪 冷 渔 翁'
    #calculate_attention_metric(epoch = 5)
    source =  u'过 天 星 似 箭'
    target_input = u'<eos> 吐 魄 月 如 弓'
    target_output = u'吐 魄 月 如 弓 <eos>'
    #target_input = u'吐 魄 月 如 弓'
    #target_output = u'吐 魄 月 如 弓'
    attention = get_attentions(
        source = source,
        target = target_input,
        epoch = 5
    )
    x_annotations = source.strip().split()
    #y_annotations = target_output.strip().split()
    outputs = target_output.strip().split()
    inputs = target_input.strip().split()
    y_annotations = []
    for i in range(len(inputs)):
        y_annotations.append(inputs[i]+' / '+outputs[i])

    draw_confusion_matrix(attention, x_annotations, y_annotations)



    