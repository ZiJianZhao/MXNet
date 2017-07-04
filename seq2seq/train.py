#-*- coding:utf-8 -*-

import mxnet as mx
import numpy as np

import re, os, sys, argparse, logging, collections
import codecs
import math
import random
from collections import defaultdict

from enc_dec_iter import EncoderDecoderIter, read_dict, get_enc_dec_text_id, generate_buckets
from seq2seq import Seq2Seq
from focus_attention import FocusSeq2Seq
from global_attention import GlobalSeq2Seq
from eval_and_visual import read_file
from metric import PerplexityWithoutExp
from nltk.translate.bleu_score import corpus_bleu

def rescore(string, results):
    res = []
    string = string.strip().split()
    for line in results:
        sent = line[1].strip().split()
        if len(sent) != len(string):
            continue
        flag = True
        for i in range(len(string)):
            if string[i] == u'，' or string[i] == u'。':
                if sent[i] != string[i]:
                    flag = False
                    break
            else:
                if sent[i] == u'，' or sent[i] == u'。':
                    flag = False
                    break
        if flag:
            res.append(line)
    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Encoder-Decoder Model Inference")
    parser.add_argument('--mode', default = 'test', type = str, 
        help='you want to train or test or generate')
    parser.add_argument('--task', default = 'sort', type = str, 
        help='which task: stc, couplet, or other')
    parser.add_argument('--model', default = 'seq2seq', type = str,
        help='which model: seq2seq, focus_attention_seq2seq, global_attention_seq2seq')
    
    parser.add_argument('--testepoch', default = '6', type = int,
        help='test epoch')
    args = parser.parse_args()
    print args
    mode = args.mode
    task = args.task
    model = args.model
    testepoch = args.testepoch
    if model == 'seq2seq':
        Model = Seq2Seq
    elif model == 'global':
        Model = GlobalSeq2Seq
    else:
        Model = FocusSeq2Seq

    # ----------------- 0. Process the data  ---------------------------------------
    # ----------------- 0.1 stc data -----------------------------------------------
    if task == 'stc':
        task_dir = '/slfs1/users/zjz17/github/data/stc_2017/'
        data_dir = task_dir + 'data/'
        params_dir = task_dir + model + '_params/'
        params_prefix = 'stc'
        enc_vocab_file = 'post.vocab'
        dec_vocab_file = 'cmnt.vocab'
        train_file = 'train.txt'
        valid_file = 'valid.txt'
        test_file = 'test.txt'
        share_embed_weight = False
    elif task == 'couplet':
        task_dir = '/slfs1/users/zjz17/github/data/couplet/'
        data_dir = task_dir + 'data/final'
        params_dir = task_dir + model + '_params/'
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
    else:
        task_dir = '/slfs1/users/zjz17/github/data/' + task + '/'
        data_dir = task_dir + 'data/'
        params_dir = task_dir + model + '_params/'
        params_prefix = task
        enc_vocab_file = 'q0.vocab'
        dec_vocab_file = 'q0.vocab'
        train_file = 'q0.train'
        valid_file = 'q0.valid'
        test_file = 'q0.valid'
        share_embed_weight = True
    enc_word2idx = read_dict(os.path.join(data_dir, enc_vocab_file))
    dec_word2idx = read_dict(os.path.join(data_dir, dec_vocab_file))
    ignore_label = dec_word2idx.get('<pad>')
    # ----------------- 1. Configure logging module  ---------------------------------------
    # This is needed only in train mode
    if mode == 'train':
        logging.basicConfig(
            level = logging.DEBUG,
            format = '%(asctime)s %(message)s', 
            datefmt = '%m-%d %H:%M:%S %p',  
            filename = task_dir + model + '_Log',
            filemode = 'w'
        )
        logger = logging.getLogger()
        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG)
        logger.addHandler(console)
    
    # -----------------2. Params Defination ----------------------------------------
    num_buckets = 4
    batch_size = 32

    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)

    def sym_gen(bucketkey):
        seq2seq = Model(
            enc_input_size = len(enc_word2idx), 
            dec_input_size = len(dec_word2idx),
            enc_len = bucketkey[0],
            dec_len = bucketkey[1],
            num_label = len(dec_word2idx),
            share_embed_weight = share_embed_weight,
            is_train = True
        )
        softmax_symbol = seq2seq.symbol_define()
        data_names = ['enc_data', 'enc_mask', 'dec_data', 'dec_mask']
        label_names = ['label']
        return (softmax_symbol, data_names, label_names)

    if mode == 'train':
        enc_train, dec_train = get_enc_dec_text_id(os.path.join(data_dir, train_file), enc_word2idx, dec_word2idx)
        enc_valid, dec_valid = get_enc_dec_text_id(os.path.join(data_dir, valid_file), enc_word2idx, dec_word2idx)

        # ----------------------3. Data Iterator Defination ---------------------
        sequence_length = []
        for i in range(len(enc_train)):
            sequence_length.append((len(enc_train[i]), len(dec_train[i])+1))
        for i in range(len(enc_valid)):
            sequence_length.append((len(enc_valid[i]), len(dec_valid[i])+1))

        buckets = generate_buckets(sequence_length, num_buckets)

        train_iter = EncoderDecoderIter(
            enc_data = enc_train, 
            dec_data = dec_train, 
            batch_size = batch_size, 
            buckets = buckets, 
            shuffle = True,
            pad = enc_word2idx.get('<pad>'), 
            eos = enc_word2idx.get('<eos>')
        )
        valid_iter = EncoderDecoderIter(
            enc_data = enc_valid, 
            dec_data = dec_valid, 
            batch_size = batch_size, 
            buckets = buckets,
            shuffle = False, 
            pad = enc_word2idx.get('<pad>'), 
            eos = enc_word2idx.get('<eos>')
        )
        frequent = train_iter.data_len / batch_size / 10 # log frequency
        # ------------------4. Load paramters if exists ------------------------------

        model_args = {}

        if os.path.isfile('%s/%s-symbol.json' % (params_dir, params_prefix)):
            filelist = os.listdir(params_dir)
            paramfilelist = []
            for f in filelist:
                if f.startswith('%s-' % params_prefix) and f.endswith('.params'):
                    paramfilelist.append( int(re.split(r'[-.]', f)[1]) )
            last_iteration = max(paramfilelist)
            print('laoding pretrained model %s%s at epoch %d' % (params_dir, params_prefix, last_iteration))
            sym, arg_params, aux_params = mx.model.load_checkpoint('%s/%s' % (params_dir, params_prefix), last_iteration)
            model_args.update({
                'arg_params' : arg_params,
                'aux_params' : aux_params,
                'begin_epoch' : last_iteration
                })


        # -----------------------5. Training ------------------------------------
        if not os.path.exists(params_dir):
            os.makedirs(params_dir)

        if num_buckets == 1:
            mod = mx.mod.Module(*sym_gen(train_iter.default_bucket_key), context = [mx.gpu(1)])
        else:
            mod = mx.mod.BucketingModule(
                sym_gen = sym_gen, 
                default_bucket_key = train_iter.default_bucket_key, 
                context = [mx.gpu(1)]
            )
        mod.fit(
            train_data = train_iter, 
            eval_data = valid_iter, 
            num_epoch = 8,
            eval_metric = PerplexityWithoutExp(ignore_label),
            epoch_end_callback = [mx.callback.do_checkpoint('%s%s' % (params_dir, params_prefix), 1)],
            batch_end_callback = [mx.callback.Speedometer(batch_size, frequent = frequent)],
            initializer = mx.init.Uniform(0.05),
            optimizer = 'adam',
            optimizer_params = {'wd': 0.0000, 'clip_gradient': 0.1},
            **model_args
            #optimizer_params = {'learning_rate':0.01, 'momentum': 0.9, 'wd': 0.0000}
        )
    elif mode == 'test':
        sym, arg_params, aux_params = mx.model.load_checkpoint('%s%s' % (params_dir, params_prefix), testepoch)
        dec_idx2word = {}
        for k, v in dec_word2idx.items():
            dec_idx2word[v] = k
        enc_idx2word = {}
        for k, v in enc_word2idx.items():
            enc_idx2word[v] = k
        while True:
            # ------------------- get input ---------------------
            enc_string = raw_input('Enter the encode sentence:\n')
            if not isinstance(enc_string, unicode):
                enc_string = unicode(enc_string, 'utf-8')
            string_list = enc_string.strip().split()
            enc_len = len(string_list)
            data = []
            for item in string_list:
                if enc_word2idx.get(item) is None:
                    data.append(enc_word2idx.get('<unk>'))
                else:
                    data.append(enc_word2idx.get(item))
            enc_data = mx.nd.array(np.array(data).reshape(1, enc_len))
            # --------------------- beam seqrch ------------------          
            seq2seq = Model(
                enc_input_size = len(enc_word2idx), 
                dec_input_size = len(dec_word2idx),
                enc_len = enc_len,
                dec_len = 1,
                num_label = len(dec_word2idx),
                share_embed_weight = share_embed_weight,
                is_train = False
            )
            input_str = ""
            enc_list = enc_data.asnumpy().reshape(-1,).tolist()
            for i in enc_list:
                input_str += " " +  enc_idx2word[int(i)]

            if task == 'couplet':
                results = seq2seq.couplet_predict(enc_data, arg_params)
                res = []
                for pair in results:
                    sent = pair[1]
                    mystr = ""
                    for idx in sent:
                        if dec_idx2word[idx] == '<eos>':
                            continue
                        mystr += " " +  dec_idx2word[idx]
                    res.append((pair[0], mystr.strip()))       
                results = rescore(input_str, res)

                print "Encode Sentence: ", input_str
                print 'Beam Search Results: '
                minmum = min(10, len(results))
                for pair in results[0:minmum]:
                    print "score : %f, %s" % (pair[0], pair[1]) 
            else:
                results = seq2seq.predict(enc_data, arg_params)
                # ---------------------- print result ------------------
                print "Encode Sentence: ", input_str
                print 'Beam Search Results: '
                minmum = min(10, len(results))
                for pair in results[0:minmum]:
                    score = pair[0]
                    sent = pair[1]
                    mystr = ""
                    for idx in sent:
                        if dec_idx2word[idx] == '<eos>':
                            continue
                        mystr += " " +  dec_idx2word[idx]
                    print "score : %f, %s" % (score, mystr)            
    else:
        enc_test, dec_test = get_enc_dec_text_id(os.path.join(data_dir, test_file), enc_word2idx, dec_word2idx)
        sequence_length = []
        for i in range(len(enc_test)):
            sequence_length.append((len(enc_test[i]), len(dec_test[i])+1))
        buckets = generate_buckets(sequence_length, num_buckets)
        test_iter = EncoderDecoderIter(
            enc_data = enc_test, 
            dec_data = dec_test, 
            batch_size = batch_size, 
            buckets = buckets, 
            shuffle = False,
            pad = enc_word2idx.get('<pad>'), 
            eos = enc_word2idx.get('<eos>')
        )
        if num_buckets == 1:
            mod = mx.mod.Module(*sym_gen(test_iter.default_bucket_key), context = [mx.gpu(0)])
        else:
            mod = mx.mod.BucketingModule(
                sym_gen = sym_gen, 
                default_bucket_key = test_iter.default_bucket_key, 
                context = [mx.gpu(0)]
            )
        epoch = 2
        sym, arg_params, aux_params = mx.model.load_checkpoint('%s%s' % (params_dir, params_prefix), epoch)
        mod.bind(data_shapes=test_iter.provide_data, label_shapes = test_iter.provide_label)
        mod.set_params(arg_params=arg_params, aux_params=aux_params)
        res  = mod.score(test_iter, PerplexityWithoutExp(ignore_label))
        for name, val in res:
            print 'Test-%s=%f' %  (name, val)

        dec_idx2word = {}
        for k, v in dec_word2idx.items():
            dec_idx2word[v] = k
        enc_idx2word = {}
        for k, v in enc_word2idx.items():
            enc_idx2word[v] = k
        g = open(model+'_generate_epoch_%d.txt' % epoch, 'w')
        
        path = os.path.join(data_dir, test_file)
        with codecs.open(path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        dic = defaultdict(str)
        for line in lines:
            line_list = line.strip().split('\t=>\t')
            if len(line_list) != 2 or len(line_list[0].strip()) != len(line_list[1].strip()) :
                continue
            dic[line_list[0].strip()] += line_list[1].strip() + '\t'
        index = 0
        for key in dic:
            enc_string = key
            dec_string = dic[key]
            string_list = enc_string.strip().split()
            enc_len = len(string_list)
            data = []
            for item in string_list:
                if enc_word2idx.get(item) is None:
                    data.append(enc_word2idx.get('<unk>'))
                else:
                    data.append(enc_word2idx.get(item))
            enc_data = mx.nd.array(np.array(data).reshape(1, enc_len))
            g.write('post: ' + enc_string.encode('utf8') + '\n')
            g.write('cmnt: ' + dec_string.encode('utf8') + '\n')
            g.write('beam search results:\n')
            seq2seq = Model(
                enc_input_size = len(enc_word2idx), 
                dec_input_size = len(dec_word2idx),
                enc_len = enc_len,
                dec_len = 1,
                num_label = len(dec_word2idx),
                share_embed_weight = share_embed_weight,
                is_train = False
            )
            # ---------------------- print result ------------------
            if task == 'couplet':
                results = seq2seq.couplet_predict(enc_data, arg_params)
                res = []
                for pair in results:
                    sent = pair[1]
                    mystr = ""
                    for idx in sent:
                        if dec_idx2word[idx] == '<eos>':
                            continue
                        mystr += " " +  dec_idx2word[idx]
                    res.append((pair[0], mystr.strip()))
                results = rescore(enc_string, res)
                minmum = min(10, len(results))
                for pair in results[0:minmum]:
                    g.write("score : %f, sentence: %s\n" % (pair[0], pair[1].encode('utf8')))
                g.write('==============================================\n')
            else:
                results = seq2seq.predict(enc_data, arg_params)
                minmum = min(10, len(results))
                for pair in results[0:minmum]:
                    score = pair[0]
                    sent = pair[1]
                    mystr = ""
                    for idx in sent:
                        if dec_idx2word[idx] == '<eos>':
                            continue
                        mystr += " " +  dec_idx2word[idx]
                    g.write("score : %f, sentence: %s\n" % (score, mystr.encode('utf8')))
                g.write('==============================================\n')
            index += 1
            if index > 500:
                break
        g.close()
        
        list_of_hypothesis, list_of_references = read_file(model+'_generate_epoch_%d.txt' % epoch)
        '''Use blue0 since there are some  sentences with only one word and the lexicon is built on top of one-word segmentation'''
        blue = corpus_bleu(list_of_references, list_of_hypothesis, 
            weights=(1,),
            smoothing_function=None
        )
        print 'Test-Blue: %f' % blue
