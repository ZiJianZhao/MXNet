import mxnet as mx
import numpy as np

import re, os, sys, argparse, logging,collections
import codecs

from text_io import read_dict, get_enc_dec_text_id
from enc_dec_iter import EncoderDecoderIter, read_dict, generate_init_states_for_rnn
from seq2seq import Seq2Seq, BeamSearch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Encoder-Decoder Model Inference")
    parser.add_argument('--mode', default = 'train', type = str, 
        help='you want to train or test')
    args = parser.parse_args()
    print args
    mode = args.mode
    # ----------------- 0. Configure logging module  ---------------------------------------
    logging.basicConfig(
        level = logging.DEBUG,
        format = '%(asctime)s %(message)s', 
        datefmt = '%m-%d %H:%M:%S %p',  
        filename = 'Log',
        filemode = 'w'
    )
    logger = logging.getLogger()
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    logger.addHandler(console)

    # ----------------- 1. Process the data  ---------------------------------------
    data_dir = '/slfs1/users/zjz17/github/data/sort'
    vocab_file = 'q1.vocab'
    train_file = 'q1.train'
    valid_file = 'q1.valid'
    enc_word2idx = read_dict(os.path.join(data_dir, vocab_file))
    dec_word2idx = read_dict(os.path.join(data_dir, vocab_file))

    # -----------------2. Params Defination ----------------------------------------
    num_buckets = 4
    batch_size = 64
    # rnn network paramters
    num_layers = 1
    enc_input_size = len(enc_word2idx)
    dec_input_size = len(dec_word2idx)
    enc_dropout = 0.0
    dec_dropout = 0.0
    output_dropout = 0.2
    num_embed = 400
    num_hidden = 800
    num_label = len(dec_word2idx)
    # encoder decoder parameters
    encoder_name = 'enc'
    encoder_mode = 'lstm'
    encoder_bi  = False
    decoder_name = 'dec'
    decoder_mode = 'lstm'
    init_states = generate_init_states_for_rnn(num_layers, encoder_name, encoder_mode, encoder_bi, batch_size, num_hidden)
    print init_states
    # training parameters
    params_dir = 'params'
    params_prefix = 'couplet'
    # program  parameters
    seed = 1
    np.random.seed(seed)

    if mode == 'train':
        # ----------------------3. Data Iterator Defination ---------------------
        train_iter = EncoderDecoderIter(
            enc_word2idx = enc_word2idx, 
            dec_word2idx = dec_word2idx, 
            filename = os.path.join(data_dir, train_file), 
            init_states = init_states,
            batch_size = batch_size, 
            num_buckets = num_buckets
        )
        valid_iter = EncoderDecoderIter(
            enc_word2idx = enc_word2idx, 
            dec_word2idx = dec_word2idx, 
            filename = os.path.join(data_dir, valid_file), 
            init_states = init_states,
            batch_size = batch_size, 
            num_buckets = num_buckets
        )
        frequent = train_iter.data_len / batch_size / 5 # log frequency

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
                enc_mode = encoder_mode, 
                enc_bi = encoder_bi,
                enc_num_layers = num_layers, 
                enc_len = bucketkey.enc_len,
                enc_input_size = enc_input_size, 
                enc_num_embed = num_embed, 
                enc_num_hidden = num_hidden,
                enc_dropout = enc_dropout, 
                enc_name = encoder_name,
                dec_mode = decoder_mode, 
                dec_num_layers = num_layers, 
                dec_len = bucketkey.dec_len,
                dec_input_size = dec_input_size, 
                dec_num_embed = num_embed, 
                dec_num_hidden = num_hidden,
                dec_num_label = num_label, 
                ignore_label = dec_word2idx.get('<PAD>'), 
                dec_dropout = dec_dropout, 
                dec_name = decoder_name,
                output_dropout = output_dropout
            )
            return seq2seq_model.get_softmax()

        model = mx.model.FeedForward(
            ctx = [mx.context.gpu(i) for i in range(1)],
            symbol = gen_sym,
            num_epoch = 20,
            optimizer = 'Adam',
            wd = 0.,
            initializer   = mx.init.Uniform(0.05),
            **model_args
        )

        if not os.path.exists(params_dir):
            os.makedirs(params_dir)

        ignore_label = dec_word2idx.get('<PAD>')
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
            return float(num) / pred.shape[0] * 100

        model.fit(
            X = train_iter,
            eval_data = valid_iter,
            eval_metric = mx.metric.np(perplexity),
            batch_end_callback = [mx.callback.Speedometer(batch_size, frequent = frequent)],
            epoch_end_callback = [mx.callback.do_checkpoint('%s/%s' % (params_dir, params_prefix), 1)])
    else:
        tmp = mx.model.FeedForward.load('%s/%s' % (params_dir, params_prefix), 10)
        arg_params = tmp.arg_params
        while True:
            beam_search = BeamSearch(
                arg_params = arg_params, 
                enc_word2idx = enc_word2idx,  
                dec_word2idx = dec_word2idx, 
                enc_string = None, 
                dec_string = None,
                enc_mode = encoder_mode, 
                enc_bi = encoder_bi, 
                enc_num_layers = num_layers,
                enc_input_size = enc_input_size,   
                enc_num_embed = num_embed, 
                enc_num_hidden = num_hidden,
                enc_dropout = enc_dropout, 
                enc_name = encoder_name,
                dec_mode = decoder_mode, 
                dec_num_layers = num_layers, 
                dec_input_size = dec_input_size, 
                dec_num_embed = num_embed, 
                dec_num_hidden = num_hidden,
                dec_num_label = num_label, 
                dec_dropout = dec_dropout, 
                dec_name = decoder_name,
                output_dropout = output_dropout
            )
            beam_search.printout()
'''
from rnn.rnn import RNN
if __name__ == '__main__':
    def encoder(
        mode, bi_directional, num_layers, enc_len, 
        enc_input_size, num_embed, num_hidden, 
        enc_dropout = 0.0, name = 'enc'):
        enc_data = mx.sym.Variable('%s_data' % name)
        enc_mask = mx.sym.Variable('%s_mask' % name)
        enc_length = mx.sym.Variable('%s_leng' % name)
        enc_embed = mx.sym.Embedding(
            data = enc_data, 
            input_dim = enc_input_size, 
            output_dim = num_embed, 
            name = '%s_embed' % name
        )
        rnn_outputs = RNN(
            data = enc_embed, #totalvec, 
            mask = enc_mask, 
            mode = mode, 
            seq_len = enc_len, 
            real_seq_len = enc_length, 
            num_layers = num_layers, 
            num_hidden = num_hidden, 
            bi_directional = bi_directional, 
            states = None, 
            cells = None, 
            dropout = enc_dropout, 
            name = name
        ).get_outputs()
        encoder_last_layer_hiddens = rnn_outputs['last_layer']
        encoder_last_time_cells = rnn_outputs['last_time']['cell']
        encoder_last_time_hiddens = rnn_outputs['last_time']['hidden']
        return mx.sym.Group(encoder_last_time_hiddens)

    enc_name = 'enc'
    batch_size = 2
    num_hidden = 5
    num_lstm_layer = 1
    enc_len = 5
    bi_directional = True
    mode = 'lstm'
    encoder_symbol = encoder(
        mode = mode,
        bi_directional = bi_directional,
        num_layers = num_lstm_layer, 
        enc_len = 5, 
        enc_input_size = 10, 
        num_embed = 5, 
        num_hidden = num_hidden, 
        enc_dropout = 0.0, 
        name = 'enc'
    )
    print 'yes'
    init_states = generate_init_states_for_rnn(num_lstm_layer, 'enc', mode, bi_directional, batch_size, num_hidden)
    enc_data_shape = [("enc_data", (batch_size, enc_len))]
    enc_mask_shape = [("enc_mask", (batch_size, enc_len))]
    enc_len_shape = [('enc_leng', (batch_size,))]
    enc_input_shapes = dict(init_states + enc_data_shape + enc_mask_shape)
    # bind the network and provide the pretrained parameters
    encoder_executor = encoder_symbol.simple_bind(ctx = mx.cpu(), **enc_input_shapes)
    for key in encoder_executor.arg_dict.keys():
        if key.endswith('weight'): 
            s1, s2 = encoder_executor.arg_dict[key].shape
            #encoder_executor.arg_dict[key][:] = np.random.rand(s1, s2)
            encoder_executor.arg_dict[key][:] = 1
        print key, np.sum(encoder_executor.arg_dict[key].asnumpy())

    # provide the input data and forward the network
    enc_data = mx.nd.array([[3, 4, 5, 2, 2], [2, 2, 1, 1, 1]])
    enc_mask = mx.nd.array([[1, 1, 0, 0, 0], [1, 1, 1, 0, 0]])
    enc_len = mx.nd.array([2, 3])
    enc_data.copyto(encoder_executor.arg_dict["enc_data"])
    enc_mask.copyto(encoder_executor.arg_dict["enc_mask"])
    enc_len.copyto(encoder_executor.arg_dict['enc_leng'])
    encoder_executor.forward()
    for i in encoder_executor.outputs:
        print i.shape
        print '================='
'''
