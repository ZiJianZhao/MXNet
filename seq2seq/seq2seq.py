import sys

import mxnet as mx

sys.path.append('..')
from rnn.rnn import RNN

class Seq2Seq(object):

    def __init__(
        self, enc_mode, enc_num_layers, enc_len,
        enc_input_size, enc_num_embed, enc_num_hidden,
        enc_dropout, enc_name,
        enc_info_size, # information parameter
        dec_mode, dec_num_layers, dec_len,
        dec_input_size, dec_num_embed, dec_num_hidden,
        dec_num_label, ignore_label, dec_dropout, dec_name,
        output_dropout,
        train = True):
        super(Seq2Seq, self).__init__()

        self.enc_mode = enc_mode
        self.enc_num_layers = enc_num_layers
        self.enc_len = enc_len
        self.enc_input_size = enc_input_size
        self.enc_num_embed = enc_num_embed
        self.enc_num_hidden = enc_num_hidden
        self.enc_dropout = enc_dropout
        self.enc_name = enc_name
        self.enc_info_size = enc_info_size
        self.dec_mode = dec_mode
        self.dec_num_layers = dec_num_layers
        self.dec_len = dec_len
        self.dec_input_size = dec_input_size
        self.dec_num_embed = dec_num_embed
        self.dec_num_hidden = dec_num_hidden
        self.dec_num_label = dec_num_label
        self.ignore_label = ignore_label
        self.dec_dropout = dec_dropout
        self.dec_name = dec_name
        self.output_dropout = output_dropout
        self.train = train


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

        # ======== add information ===============
        '''enc_info = mx.sym.Variable('%s_info' % name)
        info_embed = mx.sym.Embedding(
            data = enc_info, 
            input_dim = self.enc_info_size, 
            #weight = enc_embed_weight, 
            output_dim = num_embed, 
            name = '%s_info_embed' % name
        )
        wordvec = mx.sym.SliceChannel(
            data = enc_embed,
            num_outputs = self.enc_len,
            axis = 1,
            squeeze_axis = False
        )
        infovec = mx.sym.SliceChannel(
            data = info_embed,
            num_outputs = self.enc_len,
            axis = 1,
            squeeze_axis = False
        )
        hidden_all = []
        for i in range(0, self.enc_len):
            hidden_all.append(
                mx.sym.Concat(*[wordvec[i], infovec[i]], dim = 2)
                #wordvec[i] + infovec[i]
            )
        totalvec = mx.sym.Concat(*hidden_all, dim = 1)'''
        # ========================================
        
        rnn_outputs = RNN(
            data = enc_embed, #totalvec, 
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
        hiddens = []
        for i in range(self.enc_num_layers):
            hiddens.append(self.encoder_last_time_hiddens[i])
            hiddens.append(self.encoder_last_time_cells[i])
        return mx.sym.Group(hiddens)

    def decoder(
        self, mode, num_layers, dec_len, 
        dec_input_size, num_embed, num_hidden, 
        num_label, ignore_label, dec_dropout=0.0,
        output_dropout = 0.2, name = 'dec'):
    
        dec_embed_weight = mx.sym.Variable("%s_embed_weight" % name)
        dec_data = mx.sym.Variable('%s_data' % name)
        dec_mask = mx.sym.Variable('%s_mask' % name)

        
        cls_weight = mx.sym.Variable("%s_cls_weight" % name)
        cls_bias = mx.sym.Variable("%s_cls_bias" % name)
        
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
                real_seq_len = None, 
                num_layers = num_layers, 
                num_hidden = num_hidden, 
                bi_directional = False, 
                states = self.encoder_last_time_hiddens, 
                cells = self.encoder_last_time_cells, 
                dropout = dec_dropout, 
                name = name
            ).get_outputs()
        else:
            rnn_outputs = RNN(
                data = dec_embed, 
                mask = dec_mask, 
                mode = mode, 
                seq_len = dec_len, 
                real_seq_len = None, 
                num_layers = num_layers, 
                num_hidden = num_hidden, 
                bi_directional = False, 
                states = None, 
                cells = None, 
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
        if self.train:
            label = mx.sym.Variable('label')
            label = mx.sym.transpose(data = label)
            label = mx.sym.Reshape(data = label, shape = (-1, ))
            sm = mx.sym.SoftmaxOutput(
                data = pred, 
                label = label, 
                name = '%s_softmax' % name,
                use_ignore = True, 
                ignore_label = ignore_label
            )
        else:
            sm = mx.sym.SoftmaxOutput(data = pred, name = 'test_softmax')

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
        outputs = [sm]
        for i in range(self.dec_num_layers):
            outputs.append(self.decoder_last_time_hiddens[i])
            outputs.append(self.decoder_last_time_cells[i])
        return mx.sym.Group(outputs)

    def get_softmax(self):
        self.encoder(
            mode = self.enc_mode, 
            num_layers = self.enc_num_layers, 
            enc_len = self.enc_len, 
            enc_input_size = self.enc_input_size, 
            num_embed = self.enc_num_embed, 
            num_hidden = self.enc_num_hidden, 
            enc_dropout = self.enc_dropout, 
            name = self.enc_name
        )
        self.decoder(
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
        return self.decoder_softmax