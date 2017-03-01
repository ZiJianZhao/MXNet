import mxnet as mx
import numpy as np
import os
import matplotlib 
matplotlib.use('Agg')
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from collections import defaultdict
from lstm_inference import BeamSearch
from text_io import read_dict, DataGeneration

def process_log(filename = 'i_Log'):
    f = open(filename, 'r')
    strs = f.readlines()
    train = []
    valid = []
    for i in range(1, len(strs), 9):
        train_str = strs[i+4]
        valid_str = strs[i+8]
        train_metric = float(train_str.split('=')[1])
        valid_metric = float(valid_str.split('=')[1])
        train.append(train_metric)
        valid.append(valid_metric)
    return train, valid

def plot_multiple_lines():
    # =========== ppl without info of seq2seq ==========
    #plt.subplot(2,1,1)
    train_1, valid_1 = process_log('log_dir/Log')
    train_2, valid_2 = process_log('log_dir/with_fix_embed_Log')
    train_3, valid_3 = process_log('log_dir/with_embed_Log')
    x = range(len(train_1))
    #plt.subplot(2,1,1)
    plt.plot(x, train_1, 'r-o', label='train')
    plt.plot(x, valid_1, 'r-x', label='valid')
    plt.plot(x, train_2, 'b-o', label='train_fix_embed')
    plt.plot(x, valid_2, 'b-x', label='valid_fix_embed')
    plt.plot(x, train_3, 'g-o', label='train_embed')
    plt.plot(x, valid_3, 'g-x', label='valid_embed')    
    plt.legend(loc = 'lower right')
    plt.title('True-Metric with_or_without pretrained (fixed) word vectors') 
    plt.ylabel('True-Metric')
    '''train_1, valid_1 = process_log('log_dir/no_info_cm_Log')
    train_2, valid_2 = process_log('log_dir/info_cm_Log')    
    plt.subplot(2,1,2)
    plt.plot(x, train_1, 'r-o', label='train_no_info')
    plt.plot(x, valid_1, 'r-x', label='valid_no_info')
    plt.plot(x, train_2, 'b-o', label='train_info')
    plt.plot(x, valid_2, 'b-x', label='valid_info')
    plt.legend(loc = 'lower right')
    plt.ylabel('True-Metric')
    plt.xlabel('epoch')'''
    plt.show()
    plt.savefig('tt.jpg')

plot_multiple_lines()

def see_embed_params(params_dir = 'optimal_params', params_prefix = 'couplet', epoch = 20):
    model_args  = {}
    tmp = mx.model.FeedForward.load('%s/%s' % (params_dir, params_prefix), epoch)
    model_args.update({
        'arg_params' : tmp.arg_params,
        'aux_params' : tmp.aux_params,
        'begin_epoch' : tmp.begin_epoch
        })
    enc_embed_weight = model_args['arg_params']['embed_weight'].asnumpy()[3:, :]
    #dec_embed_weight = model_args['arg_params']['dec_embed_weight'].asnumpy()
    #print enc_embed_weight.shape
    model = TSNE(n_components=2, random_state=0, learning_rate = 500, n_iter = 2000)
    x = model.fit_transform(enc_embed_weight)
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    dic = defaultdict(list)
    for i in range(1000):
        idx = sum(map(int, str(i)))
        dic[idx].append(i)
    for i in range(10,17):
            plt.scatter(x[dic[i],1], x[dic[i],0], s=20, marker = 'o', color = colors[i-10], label='%d' % i)

    plt.legend(loc = 'upper right')
    plt.title('Word Embedding T-SNE Visualization')
    plt.savefig('ff.jpg')

#see_embed_params()

def see_hidden_vectos(params_dir = 'optimal_params', params_prefix = 'couplet', epoch = 20):
    _, arg_params, __ = mx.model.load_checkpoint('%s/%s' % (params_dir, params_prefix), epoch)
    results = []
    moban = [[5, 5, 5, 5, 5], [10, 10, 10, 10, 10],[15, 15, 15, 15, 15], [20, 20, 20, 20, 20], [25, 25, 25, 25, 25]]
    moban = [[5, 10, 15, 20, 25], [15, 15, 15, 15, 15],[10, 10, 10, 20, 25], [10, 20, 20, 20, 5], [13, 14, 15, 16, 17]]
    num = 50
    for i in range(len(moban)):
        results.append([])
    # parameter definition 
    data_dir = '/slfs1/users/zjz17/github/data/sort'
    vocab_file = 'q3.vocab'
    enc_word2idx = read_dict(os.path.join(data_dir, vocab_file))
    dec_word2idx = read_dict(os.path.join(data_dir, vocab_file))
    num_lstm_layer = 1
    num_embed = 100
    num_hidden = 200
    num_label = len(dec_word2idx)
    batch_size = 1
    enc_input_size = len(enc_word2idx)
    dec_input_size = len(dec_word2idx)
    enc_dropout = 0.0
    dec_dropout = 0.0
    output_dropout = 0.2
    dg = DataGeneration(1000,1,1,1)
    for i in range(len(moban)):
        lis = dg.generate_test_pairs(moban[i], num)
        for l in lis:
            enc_len = len(l)
            enc_data  = mx.nd.array(np.array(l).reshape(1, enc_len)+3)
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
            ctx = mx.cpu(), 
            enc_dropout = enc_dropout, 
            dec_dropout = dec_dropout,
            output_dropout = output_dropout)
            v =  beam.init_states_dict['dec_l0_init_c'].asnumpy()
            results[i].append(v)
    ff = []
    for i in range(len(moban)):
        ff.append(results[i][0])
    for i in range(1, num):
        for j in range(len(moban)):
            ff[j] = np.concatenate((ff[j], results[j][i]))
    f = np.concatenate((ff[0], ff[1]))
    for i in range(2, len(moban)):
        f = np.concatenate((f, ff[i]))
    model = TSNE(n_components=3, random_state=0, learning_rate = 500, n_iter = 2000)
    x = model.fit_transform(f)
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i in range(len(moban)):
        tmp = range(i*50, (i+1)*50)
        plt.scatter(x[tmp,1], x[tmp,0], s=20, marker = 'o', color = colors[i], label='%s' % moban[i])
    
    ''' Three-dimension Graph
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = Axes3D(fig)
    for i in range(len(moban)):
        tmp = range(i*50, (i+1)*50)
        ax.scatter(x[tmp,0], x[tmp,1], x[tmp,2], s=20, marker = 'o', color = colors[i], label='%s' % moban[i])'''    
    plt.legend(loc = 'upper left')
    plt.title('Encoded Hidden Vector T-SNE Visualization')
    plt.savefig('ff.jpg')

#see_hidden_vectos()
