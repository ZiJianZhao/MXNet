import sys

import mxnet as mx

sys.path.append('..')
from rnn.rnn import RNN

class Seq2Seq():

    def __init__(
        self, enc_mode, enc_num_layers, enc_len,
        enc_input_size, enc_num_embed, enc_num_hidden,
        enc_dropout, enc_name,
        dec_mode, dec_num_layers, dec_len,
        dec_input_size, dec_num_embed, dec_num_hidden,
        dec_num_label, ignore_label, dec_dropout, dec_name,
        output_dropout):


        self.encoder(
            mode = enc_mode, 
            num_layers = enc_num_layers, 
            enc_len = enc_len, 
            enc_input_size = enc_input_size, 
            num_embed = enc_num_embed, 
            num_hidden = enc_num_hidden, 
            enc_dropout = enc_dropout, 
            name = enc_name
        )
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


    def encoder(
        self, mode, num_layers, enc_len, 
        enc_input_size, num_embed, num_hidden, 
        enc_dropout = 0.0, name = 'enc'):

        enc_embed_weight=mx.sym.Variable("%s_embed_weight" % name)
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
            data = enc_embed, 
            mask = enc_mask, 
            mode = mode, 
            seq_len = enc_len, 
            real_seq_len = None, 
            num_layers = num_layers, 
            num_hidden = num_hidden, 
            bi_directional = False, 
            states = None, 
            cells = None, 
            dropout = enc_dropout, 
            name = name
        ).get_outputs()
        
        self.encoder_last_layer_hiddens = rnn_outputs['last_layer']
        self.encoder_last_time = rnn_outputs['last_time']

        self.encoder_last_time_hiddens = []
        self.encoder_last_time_cells = []
        for i in range(2*num_layers):
            symbol = self.encoder_last_time[i]
            if i % 2 == 0:
                self.encoder_last_time_hiddens.append(symbol)
            else:
                self.encoder_last_time_cells.append(symbol)

    def decoder(
        self, mode, num_layers, dec_len, 
        dec_input_size, num_embed, num_hidden, 
        num_label, ignore_label, dec_dropout=0.0,
        output_dropout = 0.2, name = 'dec'):
    
        dec_embed_weight = mx.sym.Variable("%s_embed_weight" % name)
        dec_data = mx.sym.Variable('%s_data' % name)
        dec_mask = mx.sym.Variable('%s_mask' % name)

        label = mx.sym.Variable('label')
        cls_weight = mx.sym.Variable("%s_cls_weight" % name)
        cls_bias = mx.sym.Variable("%s_cls_bias" % name)
        
        dec_embed = mx.sym.Embedding(
            data = dec_data, 
            input_dim = dec_input_size, 
            weight = dec_embed_weight, 
            output_dim = num_embed, 
            name = '%s_embed' % name
        )

        rnn_outputs = RNN(
            data = dec_embed, 
            mask = dec_mask, 
            mode = mode, 
            seq_len = dec_len, 
            real_seq_len = None, 
            num_layers = num_layers, 
            num_hidden = num_hidden, 
            bi_directional = False, 
            states = self.encoder_last_time_hiddens, 
            cells = self.encoder_last_time_cells, 
            dropout = dec_dropout, 
            name = name
        ).get_outputs()

        dec_hidden_all = rnn_outputs['last_layer']
        hidden_concat = mx.sym.Concat(*dec_hidden_all, dim = 0)
        hidden_concat = mx.sym.Dropout(data=hidden_concat, p=output_dropout)
        pred = mx.sym.FullyConnected(
            data = hidden_concat, 
            num_hidden = num_label, 
            weight = cls_weight, 
            bias = cls_bias, 
            name = '%s_pred' % name
        )
        #pred = mx.sym.Reshape(data = pred, shape = (-1, dec_len, num_label))
        label = mx.sym.transpose(data = label)
        label = mx.sym.Reshape(data = label, shape = (-1, ))
        sm = mx.sym.SoftmaxOutput(
            data = pred, 
            label = label, 
            name = '%s_softmax' % name,
            use_ignore = True, 
            ignore_label = ignore_label
        )
        self.decoder_last_layer_hiddens = rnn_outputs['last_layer']
        self.decoder_last_time = rnn_outputs['last_time']
        self.decoder_last_time_hiddens = []
        self.decoder_last_time_cells = []
        for i in range(len(self.decoder_last_time)):
            symbol = self.decoder_last_time[i]
            if i % 2 == 0:
                self.decoder_last_time_hiddens.append(symbol)
            else:
                self.decoder_last_time_cells.append(symbol)
        self.decoder_softmax = sm

    def get_softmax(self):
        return self.decoder_softmax