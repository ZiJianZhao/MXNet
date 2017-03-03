import sys
import copy
import math
import mxnet as mx
import numpy as np 
sys.path.append('..')
from rnn.rnn import RNN
from enc_dec_iter import generate_init_states_for_rnn

class Seq2Seq(object):
    '''Sequence to sequence learning with neural networks
    The basic sequence to sequence learning network

    Note: you can't use gru as encoder and lstm as decoder
    because so makes the lstm cell has no initilization. 
    '''

    def __init__(
        self, enc_mode, enc_bi, enc_num_layers, enc_len,
        enc_input_size, enc_num_embed, enc_num_hidden,
        enc_dropout, enc_name,
        dec_mode, dec_num_layers, dec_len,
        dec_input_size, dec_num_embed, dec_num_hidden,
        dec_num_label, ignore_label, dec_dropout, dec_name,
        output_dropout, share_embed_weight = True, train = True):
        super(Seq2Seq, self).__init__()
        self.train = train
        if share_embed_weight:  # (for same language task, for example, dialog)
            self.embed_weight = mx.sym.Variable('embed_weight')
            self.enc_embed_weight = self.embed_weight
            self.dec_embed_weight = self.embed_weight
        else:  # (for multi languages task, for example, translation)
            self.enc_embed_weight = mx.sym.Variable('%s_embed_weight' % enc_name)
            self.dec_embed_weight = mx.sym.Variable('%s_embed_weight' % dec_name)
        
        self.encoder(
            mode = enc_mode, 
            bi_directional = enc_bi,
            num_layers = enc_num_layers, 
            enc_len = enc_len, 
            enc_input_size = enc_input_size, 
            num_embed = enc_num_embed, 
            num_hidden = enc_num_hidden, 
            enc_dropout = enc_dropout, 
            name = enc_name
        )
        self.decoder_init_time_cells = self.encoded_states_tranform(self.encoder_last_time_cells, dec_num_hidden, 'cell')
        self.decoder_init_time_hiddens = self.encoded_states_tranform(self.encoder_last_time_hiddens, dec_num_hidden, 'hidden')

        self.decoder(
            mode = dec_mode, 
            num_layers = dec_num_layers, 
            dec_len = dec_len, 
            dec_input_size = dec_input_size, 
            num_embed = dec_num_embed, 
            num_hidden = dec_num_hidden, 
            num_label = dec_num_label, 
            ignore_label = ignore_label, 
            dec_dropout = dec_dropout,
            output_dropout = output_dropout, 
            name = dec_name
        )

    def encoded_states_tranform(self, init_states_list, decoder_num_hidden, vector_type):
        '''
        Transform the encoded vector into decoder initial vectors
        if the vector_type is hidden, then the activation function is applied
        elif the vector_type is cell, then the activation function will not be applied
        else, raise Error
        '''
        if init_states_list is None:
            return None
        flag = True
        if vector_type == 'hidden':
            flag = True
        elif vector_type == 'cell':
            flag = False
        else:
            raise Exception("Invalid vector type parameter in function encoded_states_tranform")
        trans_weight = mx.sym.Variable("encode_to_decode_%s_trans_weight" % vector_type)
        trans_bias = mx.sym.Variable("encode_to_decode_%s_trans_bias" % vector_type)
        result_init_states_list = []
        for symbol in init_states_list:
            transformed = mx.sym.FullyConnected(
                data = symbol,
                weight = trans_weight,
                bias = trans_bias,
                num_hidden = decoder_num_hidden,
                name = "t%d_l%d_gates_i2h"
            )
            if flag:
                transformed = mx.sym.Activation(transformed, act_type = "tanh")
            result_init_states_list.append(transformed)
        return result_init_states_list


    def encoder(
        self, mode, bi_directional, num_layers, enc_len, 
        enc_input_size, num_embed, num_hidden, 
        enc_dropout = 0.0, name = 'enc'):
        enc_embed_weight = self.enc_embed_weight
        enc_data = mx.sym.Variable('%s_data' % name)
        enc_mask = mx.sym.Variable('%s_mask' % name)
        enc_embed = mx.sym.Embedding(
            data = enc_data, 
            input_dim = enc_input_size, 
            weight = enc_embed_weight, 
            output_dim = num_embed, 
            name = '%s_embed' % name
        )
        rnn_outputs = RNN(
            data = enc_embed, #totalvec, 
            mask = enc_mask, 
            mode = mode, 
            seq_len = enc_len, 
            num_layers = num_layers, 
            num_hidden = num_hidden, 
            bi_directional = bi_directional, 
            states = None, 
            cells = None, 
            dropout = enc_dropout, 
            name = name
        ).get_outputs()
        self.encoder_last_layer_hiddens = rnn_outputs['last_layer']
        self.encoder_last_time_cells = rnn_outputs['last_time']['cell']
        self.encoder_last_time_hiddens = rnn_outputs['last_time']['hidden']

    def decoder(
        self, mode, num_layers, dec_len, 
        dec_input_size, num_embed, num_hidden, 
        num_label, ignore_label, dec_dropout=0.0,
        output_dropout = 0.2, name = 'dec'):
    
        dec_embed_weight = self.dec_embed_weight
        dec_data = mx.sym.Variable('%s_data' % name)
        dec_mask = mx.sym.Variable('%s_mask' % name)
        
        dec_embed = mx.sym.Embedding(
            data = dec_data, 
            input_dim = dec_input_size, 
            weight = dec_embed_weight, 
            output_dim = num_embed, 
            name = '%s_embed' % name
        )
        if self.train:
            rnn_outputs = RNN(
                data = dec_embed, 
                mask = dec_mask, 
                mode = mode, 
                seq_len = dec_len, 
                num_layers = num_layers, 
                num_hidden = num_hidden, 
                bi_directional = False, 
                states = self.decoder_init_time_hiddens, 
                cells = self.decoder_init_time_cells, 
                dropout = dec_dropout, 
                name = name
            ).get_outputs()
        else:
            rnn_outputs = RNN(
                data = dec_embed, 
                mask = dec_mask, 
                mode = mode, 
                seq_len = dec_len, 
                num_layers = num_layers, 
                num_hidden = num_hidden, 
                bi_directional = False, 
                states = None, 
                cells = None, 
                dropout = dec_dropout, 
                name = name
            ).get_outputs()            

        self.decoder_last_layer_hiddens = rnn_outputs['last_layer']
        self.decoder_last_time_cells = rnn_outputs['last_time']['cell']
        self.decoder_last_time_hiddens = rnn_outputs['last_time']['hidden'] 

        hidden_concat = mx.sym.Concat(*self.decoder_last_layer_hiddens, dim = 0)
        hidden_concat = mx.sym.Dropout(data=hidden_concat, p=output_dropout)
        pred = mx.sym.FullyConnected(
            data = hidden_concat, 
            num_hidden = num_label,
            name = '%s_pred' % name
        )
        label = mx.sym.Variable('label')
        label = mx.sym.transpose(data = label)
        label = mx.sym.Reshape(data = label, shape = (-1, ))
        if self.train:
            sm = mx.sym.SoftmaxOutput(
                data = pred, 
                label = label, 
                name = '%s_softmax' % name,
                use_ignore = True, 
                ignore_label = ignore_label
            )
        else:
            sm = mx.sym.SoftmaxOutput(
                data = pred, 
                name = '%s_softmax' % name,
            )            
        self.sm = sm

    def get_softmax(self):
        return self.sm

class BeamSearch(Seq2Seq):

    def __init__(
        self, arg_params, enc_word2idx,  dec_word2idx, enc_string, dec_string,
        enc_mode, enc_bi, enc_num_layers,
        enc_input_size, enc_num_embed, enc_num_hidden,
        enc_dropout, enc_name,
        dec_mode, dec_num_layers, 
        dec_input_size, dec_num_embed, dec_num_hidden,
        dec_num_label, dec_dropout, dec_name,
        output_dropout, share_embed_weight = True, ctx = mx.cpu()
        ):
        # process input encoder string
        if enc_string is None:
            print 'Enter the encode sentence:'
            enc_string = raw_input()
        string_list = enc_string.strip().split(' ')
        enc_len = len(string_list)
        data = []
        for item in string_list:
            if enc_word2idx.get(item) is None:
                data.append(enc_word2idx.get('<UNK>'))
            else:
                data.append(enc_word2idx.get(item))
        enc_data = mx.nd.array(np.array(data).reshape(1, enc_len))
        enc_mask = mx.nd.array(np.ones(enc_len,).reshape(1, enc_len))
        self.enc_data = enc_data
        self.dec_string = dec_string
        self.enc_word2idx = enc_word2idx
        self.dec_word2idx = dec_word2idx
        self.batch_size = 1
        self.eos = self.dec_word2idx.get('<EOS>')
        self.unk = self.dec_word2idx.get('<UNK>')
        self.pad = self.dec_word2idx.get('<PAD>')

        # proceed encoder and decoder
        super(BeamSearch, self).__init__(
            enc_mode = enc_mode, 
            enc_bi = enc_bi,
            enc_num_layers = enc_num_layers, 
            enc_len = enc_len,
            enc_input_size = enc_input_size, 
            enc_num_embed = enc_num_embed, 
            enc_num_hidden = enc_num_hidden,
            enc_dropout = enc_dropout, 
            enc_name = enc_name,
            dec_mode = dec_mode, 
            dec_num_layers = dec_num_layers, 
            dec_len = 1,
            dec_input_size = dec_input_size, 
            dec_num_embed = dec_num_embed, 
            dec_num_hidden = dec_num_hidden,
            dec_num_label = dec_num_label, 
            ignore_label = -1, 
            dec_dropout = dec_dropout, 
            dec_name = dec_name,
            output_dropout = output_dropout,
            share_embed_weight = True,
            train = False
        )

        self.decoder_init_states = self.decoder_init_time_hiddens
        if self.decoder_init_time_cells is not None:
            self.decoder_init_states += self.decoder_init_time_cells
        test_encoder = mx.sym.Group(self.decoder_init_states)

        self.decoder_last_states = self.decoder_last_time_hiddens
        if self.decoder_last_time_cells is not None:
            self.decoder_last_states += self.decoder_last_time_cells
        test_decoder = mx.sym.Group([self.sm] + self.decoder_last_states)

        enc_init_states = generate_init_states_for_rnn(enc_num_layers, enc_name, enc_mode, enc_bi, self.batch_size, enc_num_hidden)
        enc_data_shape = [("enc_data", (self.batch_size, enc_len))]
        enc_mask_shape = [("enc_mask", (self.batch_size, enc_len))]
        enc_input_shapes = dict(enc_init_states + enc_data_shape + enc_mask_shape)
        encoder_executor = test_encoder.simple_bind(ctx = ctx, **enc_input_shapes)
        for key in encoder_executor.arg_dict.keys():
            if key in arg_params:
                arg_params[key].copyto(encoder_executor.arg_dict[key])
        # provide the input data and forward the network
        enc_data.copyto(encoder_executor.arg_dict["enc_data"])
        enc_mask.copyto(encoder_executor.arg_dict["enc_mask"]) 
        encoder_executor.forward()

        # ---------------------------- 2. Define the decoder inference model ------------------
        dec_init_states = generate_init_states_for_rnn(dec_num_layers, dec_name, dec_mode, False, self.batch_size, dec_num_hidden)
        self.init_states_dict = {}
        self.state_name = []
        for i in range(len(dec_init_states)):
            self.init_states_dict[dec_init_states[i][0]] = encoder_executor.outputs[i]
            self.state_name.append(dec_init_states[i][0])
        dec_data_shape = [("dec_data", (self.batch_size,1))]
        dec_mask_shape = [("dec_mask", (self.batch_size,1))]
        dec_input_shapes = dict(dec_init_states + dec_data_shape + dec_mask_shape)
        self.decoder_executor = test_decoder.simple_bind(ctx = ctx, **dec_input_shapes)
        for key in self.decoder_executor.arg_dict.keys():
            if key in arg_params:
                arg_params[key].copyto(self.decoder_executor.arg_dict[key])

        self.dec_data = mx.nd.zeros((self.batch_size,1))
        self.dec_mask = mx.nd.ones((self.batch_size,1))

    def printout(self):
        self.dec_idx2word = {}
        for k, v in self.dec_word2idx.items():
            self.dec_idx2word[v] = k

        self.enc_idx2word = {}
        for k, v in self.enc_word2idx.items():
            self.enc_idx2word[v] = k

        input_str = ""
        enc_list = self.enc_data.asnumpy().reshape(-1,).tolist()
        for i in enc_list:
            input_str += " " +  self.enc_idx2word[int(i)]
        print "Encode Sentence: ", input_str
        print "Decode Sentence: ", self.dec_string
        print 'Beam Search Results: '
        result_sentences = self.beam_search()
        for pair in result_sentences:
            score = pair[0]
            sent = pair[1]
            mystr = ""
            for idx in sent:
                if self.dec_idx2word[idx] == '<EOS>':
                    continue
                mystr += " " +  self.dec_idx2word[idx]
            print "score : %f, %s" % (score, mystr)


    def beam_search(self):
        self.beam = 10
        active_sentences = [(0,[self.eos], copy.deepcopy(self.init_states_dict))]
        ended_sentences = []
        min_length = 0
        max_length = 30
        min_count = min(self.beam, len(active_sentences))
        for seqidx in xrange(max_length):
            tmp_sentences = []
            for i in xrange(min_count):
                states_dict  = active_sentences[i][2]
                for key in states_dict.keys():
                    states_dict[key].copyto(self.decoder_executor.arg_dict[key])
                self.dec_data[:] = active_sentences[i][1][-1]
                self.dec_data.copyto(self.decoder_executor.arg_dict["dec_data"])
                self.dec_mask.copyto(self.decoder_executor.arg_dict["dec_mask"])
                self.decoder_executor.forward()
                new_states_dict = dict(zip(self.state_name, self.decoder_executor.outputs[1:]))
                tmp_states_dict = copy.deepcopy(new_states_dict)

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
                        tmp_sentences.append((score, sent, tmp_states_dict))

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