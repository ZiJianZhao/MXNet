import mxnet as mx
import numpy as np

import re, os, sys, argparse, logging,collections
import codecs
import math

from enc_dec_iter import EncoderDecoderIter, read_dict
from seq2seq import Seq2Seq


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Encoder-Decoder Model Inference")
    parser.add_argument('--mode', default = 'train', type = str, 
        help='you want to train or test or generate')
    parser.add_argument('--task', default = 'sort', type = str, 
        help='which task: stc, couplet, or other')    
    args = parser.parse_args()
    print args
    mode = args.mode
    task = args.task

    # ----------------- 0. Process the data  ---------------------------------------
    # ----------------- 0.1 stc data -----------------------------------------------
    if task == 'stc':
        task_dir = '/slfs1/users/zjz17/github/data/stc_2017/'
        data_dir = task_dir + 'data/'
        params_dir = task_dir + 'params/'
        params_prefix = 'stc'
        enc_vocab_file = 'post.vocab'
        dec_vocab_file = 'cmnt.vocab'
        train_file = 'train.txt'
        valid_file = 'valid.txt'
        test_file = 'test.txt'
        share_embed_weight = False
    elif task == 'couplet':
        task_dir = '/slfs1/users/zjz17/github/data/couplet/'
        data_dir = task_dir + 'data/'
        params_dir = task_dir + 'params/'
        params_prefix = 'couplet'
        enc_vocab_file = 'shanglist.txt'
        dec_vocab_file = 'xialist.txt'
        train_file = 'all.txt'
        valid_file = 'valid.txt'
        test_file = 'valid.txt'
        share_embed_weight = False
    else:
        task_dir = '/slfs1/users/zjz17/github/data/' + task + '/'
        data_dir = task_dir + 'data/'
        params_dir = task_dir + 'params/'
        params_prefix = task
        enc_vocab_file = 'q1.vocab'
        dec_vocab_file = 'q1.vocab'
        train_file = 'q1.train'
        valid_file = 'q1.valid'
        test_file = 'q1.valid'
        share_embed_weight = True
    enc_word2idx = read_dict(os.path.join(data_dir, enc_vocab_file))
    dec_word2idx = read_dict(os.path.join(data_dir, dec_vocab_file))
    # ----------------- 1. Configure logging module  ---------------------------------------
    # This is needed only in train mode
    if mode == 'train':
        logging.basicConfig(
            level = logging.DEBUG,
            format = '%(asctime)s %(message)s', 
            datefmt = '%m-%d %H:%M:%S %p',  
            filename = task_dir + 'Log',
            filemode = 'a'
        )
        logger = logging.getLogger()
        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG)
        logger.addHandler(console)
    
    # -----------------2. Params Defination ----------------------------------------
    num_buckets = 5
    batch_size = 64

    seed = 1
    np.random.seed(seed)

    if mode == 'train':
        # ----------------------3. Data Iterator Defination ---------------------
        train_iter = EncoderDecoderIter(
            enc_word2idx = enc_word2idx, 
            dec_word2idx = dec_word2idx, 
            filename = os.path.join(data_dir, train_file), 
            batch_size = batch_size, 
            num_buckets = num_buckets
        )
        valid_iter = EncoderDecoderIter(
            enc_word2idx = enc_word2idx, 
            dec_word2idx = dec_word2idx, 
            filename = os.path.join(data_dir, valid_file), 
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
            print('laoding pretrained model %s%s at epoch %d' % (params_dir, params_prefix, last_iteration))
            sym, arg_params, aux_params = mx.model.load_checkpoint('%s/%s' % (params_dir, params_prefix), last_iteration)
            model_args.update({
                'arg_params' : arg_params,
                'aux_params' : aux_params,
                'begin_epoch' : last_iteration
                })


        # -----------------------5. Training ------------------------------------
        def sym_gen(bucketkey):
            seq2seq = Seq2Seq(
                enc_input_size = len(enc_word2idx), 
                dec_input_size = len(dec_word2idx),
                enc_len = bucketkey.enc_len,
                dec_len = bucketkey.dec_len,
                num_label = len(dec_word2idx),
                share_embed_weight = share_embed_weight,
                is_train = True
            )
            softmax_symbol = seq2seq.symbol_define()
            data_names = [item[0] for item in train_iter.provide_data]
            label_names = [item[0] for item in train_iter.provide_label]
            return (softmax_symbol, data_names, label_names)

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

        if num_buckets == 1:
            mod = mx.mod.Module(*sym_gen(train_iter.default_bucket_key), context = [mx.gpu(0)])
        else:
            mod = mx.mod.BucketingModule(
                sym_gen = sym_gen, 
                default_bucket_key = train_iter.default_bucket_key, 
                context = [mx.gpu(0)]
            )
        mod.fit(
            train_data = train_iter, 
            eval_data = valid_iter, 
            num_epoch = 30,
            eval_metric = mx.metric.np(perplexity),
            epoch_end_callback = [mx.callback.do_checkpoint('%s%s' % (params_dir, params_prefix), 1)],
            batch_end_callback = [mx.callback.Speedometer(batch_size, frequent = frequent)],
            initializer = mx.init.Uniform(0.05),
            optimizer = 'Adam',
            optimizer_params = {'wd': 0.0000},
            **model_args
            #optimizer_params = {'learning_rate':0.01, 'momentum': 0.9, 'wd': 0.0000}
        )
    elif mode == 'test':
        sym, arg_params, aux_params = mx.model.load_checkpoint('%s%s' % (params_dir, params_prefix), 8)
        print arg_params
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
                    data.append(enc_word2idx.get('<UNK>'))
                else:
                    data.append(enc_word2idx.get(item))
            enc_data = mx.nd.array(np.array(data).reshape(1, enc_len))
            # --------------------- beam seqrch ------------------          
            seq2seq = Seq2Seq(
                enc_input_size = len(enc_word2idx), 
                dec_input_size = len(dec_word2idx),
                enc_len = enc_len,
                dec_len = 1,
                num_label = len(dec_word2idx),
                share_embed_weight = share_embed_weight,
                is_train = False
            )
            results = seq2seq.predict(enc_data, arg_params)
            # ---------------------- print result ------------------
            input_str = ""
            enc_list = enc_data.asnumpy().reshape(-1,).tolist()
            for i in enc_list:
                input_str += " " +  enc_idx2word[int(i)]
            print "Encode Sentence: ", input_str
            print 'Beam Search Results: '
            for pair in results:
                score = pair[0]
                sent = pair[1]
                mystr = ""
                for idx in sent:
                    if dec_idx2word[idx] == '<EOS>':
                        continue
                    mystr += " " +  dec_idx2word[idx]
                print "score : %f, %s" % (score, mystr)            
    else:
        dec_idx2word = {}
        for k, v in dec_word2idx.items():
            dec_idx2word[v] = k
        enc_idx2word = {}
        for k, v in enc_word2idx.items():
            enc_idx2word[v] = k
        g = open(task_dir+'generate.txt', 'w')
        sym, arg_params, aux_params = mx.model.load_checkpoint('%s%s' % (params_dir, params_prefix), 7)
        path = os.path.join(data_dir, test_file)
        with codecs.open(path, 'r', encoding='utf-8', errors='ignore') as f:
            enc_string = None
            dec_string = None
            while True:
                line = f.readline()
                if line == '':
                    break
                line_list = line.strip().split('\t=>\t')
                if len(line_list) == 0 or line_list[0] == enc_string:
                    continue
                enc_string = line_list[0]
                string_list = enc_string.strip().split()
                enc_len = len(string_list)
                data = []
                for item in string_list:
                    if enc_word2idx.get(item) is None:
                        data.append(enc_word2idx.get('<UNK>'))
                    else:
                        data.append(enc_word2idx.get(item))
                enc_data = mx.nd.array(np.array(data).reshape(1, enc_len))
                if len(line_list) >= 2:
                    dec_string = '\n\t'.join(line_list[1:])
                g.write('post: ' + enc_string.encode('utf8') + '\n')
                g.write('cmnt: ' + dec_string.encode('utf8') + '\n')
                g.write('beam search results:\n')

                seq2seq = Seq2Seq(
                    enc_input_size = len(enc_word2idx), 
                    dec_input_size = len(dec_word2idx),
                    enc_len = enc_len,
                    dec_len = 1,
                    num_label = len(dec_word2idx),
                    share_embed_weight = share_embed_weight,
                    is_train = False
                )
                results = seq2seq.predict(enc_data, arg_params)
                # ---------------------- print result ------------------
                for pair in results:
                    score = pair[0]
                    sent = pair[1]
                    mystr = ""
                    for idx in sent:
                        if dec_idx2word[idx] == '<EOS>':
                            continue
                        mystr += " " +  dec_idx2word[idx]
                    g.write("score : %f, sentence: %s\n" % (score, mystr.encode('utf8')))
                g.write('==============================================\n')
            g.close()