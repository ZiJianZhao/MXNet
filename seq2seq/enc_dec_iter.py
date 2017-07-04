import logging
import codecs
import re
import random
from collections import namedtuple

import mxnet as mx
import numpy as np
from sklearn.cluster import KMeans

def generate_buckets(enc_dec_data, num_buckets):
    enc_dec_data = np.array(enc_dec_data)
    kmeans = KMeans(n_clusters = num_buckets, random_state = 1) # use clustering to decide the buckets
    assignments = kmeans.fit_predict(enc_dec_data) # get the assignments
    # get the max of every cluster
    clusters = np.array([np.max( enc_dec_data[assignments==i], axis=0 ) for i in range(num_buckets)])

    buckets = []
    for i in xrange(num_buckets):
        buckets.append((clusters[i][0], clusters[i][1]))
    return buckets 

def read_dict(path):
    word2idx = {'<pad>' : 0, '<eos>' : 1, '<unk>' : 2}
    idx = 3
    with codecs.open(path, 'r', encoding='utf-8', errors='ignore') as fid:
        for line in fid:
            line = line.strip(' ').strip('\n')
            if len(line) == 0:
                continue
            if word2idx.get(line) == None:
                word2idx[line] = idx
            idx += 1
    return word2idx


def get_enc_dec_text_id(path, enc_word2idx, dec_word2idx):
    enc_data = []
    dec_data = []
    white_spaces = re.compile(r'[ \n\r\t]+')
    index = 0
    with codecs.open(path, 'r', encoding='utf-8', errors='ignore') as fid:
        for line in fid.readlines():
            line = line.strip()
            line_list = line.split('\t=>\t')
            length = len(line_list)
            for i in xrange(1, length):
                enc_list = line_list[0].strip().split()
                dec_list = line_list[i].strip().split()
                enc = [enc_word2idx.get(word) if enc_word2idx.get(word) is not None else enc_word2idx.get('<unk>') for word in enc_list]
                dec = [dec_word2idx.get(word) if dec_word2idx.get(word) is not None else  dec_word2idx.get('<unk>') for word in dec_list]
                enc_data.append(enc)
                dec_data.append(dec)
                if index == 0:
                    print 'Text2digit Preprocess Example:'
                    print line_list[0].strip().encode('utf-8'), '\t=>\t', line_list[1].strip().encode('utf-8')
                    print enc, '\t=>\t', dec
                index += 1
    return enc_data, dec_data



class EncoderDecoderIter(mx.io.DataIter):
    """This iterator is specially defined for the Couplet Generation
    
    """
    def __init__(self, enc_data, dec_data, batch_size, buckets, shuffle=True, pad=0, eos=1):
        
        super(EncoderDecoderIter, self).__init__()
        # initilization
        self.enc_data = enc_data
        self.dec_data = dec_data
        self.data_len = len(self.enc_data)
        self.pad = pad
        self.eos = eos
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.data_names = ['enc_data', 'enc_mask', 'dec_data', 'dec_mask']
        self.label_names = ['label']
        # process buckets
        self.buckets = sorted(buckets)
        enc_len = max([bucket[0] for bucket in self.buckets])
        dec_len = max([bucket[1] for bucket in self.buckets])
        self.default_bucket_key = (enc_len, dec_len)
        self.assignments = []
        for idx in range(self.data_len):
            for bkt in range(len(self.buckets)):
                if len(self.enc_data[idx]) <= self.buckets[bkt][0] and len(self.dec_data[idx]) <= self.buckets[bkt][1]:
                    break
            self.assignments.append(bkt)
        buckets_count = [0 for i in range(len(self.buckets))]
        for idx in self.assignments:
            buckets_count[idx] += 1
        print 'buckets: ', self.buckets
        print 'buckets_count: ', buckets_count
        print 'default_bucket_key: ', self.default_bucket_key
        
        # generate the data , mask, label numpy array
        self.enc_data, self.enc_mask, self.dec_data, self.dec_mask, self.label = self.make_numpy_array()

        # make a random data iteration plan
        self.plan = []
        for (i, buck) in enumerate(self.enc_data):
            self.plan.extend([(i,j) for j in range(0, buck.shape[0] - batch_size + 1, batch_size)])
        if self.shuffle:
            self.idx = [np.random.permutation(x.shape[0]) for x in self.enc_data]
        else:
            self.idx = [np.arange(x.shape[0]) for x in self.enc_data]
        self.curr_plan = 0
        self.reset()
    
    def reset(self):
        self.curr_plan = 0
        if self.shuffle:
            random.shuffle(self.plan)
            for idx in self.idx:
                np.random.shuffle(idx)   

    def next(self):
        if self.curr_plan == len(self.plan):
            raise StopIteration
        i, j = self.plan[self.curr_plan]
        self.curr_plan += 1
        index = self.idx[i][j:j+self.batch_size] 

        enc_data = mx.nd.array(self.enc_data[i][index])
        enc_mask = mx.nd.array(self.enc_mask[i][index])
        dec_data = mx.nd.array(self.dec_data[i][index])
        dec_mask = mx.nd.array(self.dec_mask[i][index])      
        label = mx.nd.array(self.label[i][index])

        data_all = [enc_data, enc_mask, dec_data, dec_mask]
        label_all = [label]

        data_names = self.data_names
        label_names = self.label_names

        return mx.io.DataBatch(
            data = data_all,
            label = label_all,
            bucket_key = self.buckets[i],
            provide_data = zip(data_names, [data.shape for data in data_all]),
            provide_label = zip(label_names, [label.shape for label in label_all])
        )

    @property
    def provide_data(self):
        return [('enc_data' , (self.batch_size, self.default_bucket_key[0])),
                  ('enc_mask' , (self.batch_size, self.default_bucket_key[0])),
                  ('dec_data' , (self.batch_size, self.default_bucket_key[1])),
                  ('dec_mask' , (self.batch_size, self.default_bucket_key[1]))] 

    @property
    def provide_label(self):
        return [('label', (self.batch_size, self.default_bucket_key[1]))]
        
    def make_data_line(self, i, bucket):
        data = self.enc_data[i]
        label = self.dec_data[i]
        enc_len = bucket[0]
        dec_len = bucket[1]
        ed = np.full(enc_len, self.pad, dtype = float)
        dd = np.full(dec_len, self.pad, dtype = float)
        em = np.zeros(enc_len, dtype = float)
        dm = np.zeros(dec_len, dtype = float)      
        l  = np.full(dec_len, self.pad, dtype = float)
        
        #ed[enc_len-len(data):enc_len] = data
        #em[enc_len-len(data):enc_len] = 1.0
        ed[0:len(data)] = data
        em[0:len(data)] = 1.0
        dd[0] = self.eos
        dd[1:len(label)+1] = label 
        dm[0:len(label)+1] = 1.0
        l[0:len(label)] = label
        l[len(label)] = self.eos
        return ed, em, dd, dm, l

    def make_numpy_array(self):
        enc_data = [[] for _ in self.buckets]
        enc_mask = [[] for _ in self.buckets]
        dec_data = [[] for _ in self.buckets]
        dec_mask = [[] for _ in self.buckets]
        label  = [[] for _ in self.buckets]

        for i in xrange(self.data_len):
            bkt_idx = self.assignments[i]
            ed, em, dd, dm, l = self.make_data_line(i, self.buckets[bkt_idx])
            enc_data[bkt_idx].append(ed)
            enc_mask[bkt_idx].append(em)
            dec_data[bkt_idx].append(dd)
            dec_mask[bkt_idx].append(dm)
            label[bkt_idx].append(l)
        enc_data = [np.asarray(i, dtype='float32') for i in enc_data]
        enc_mask  = [np.asarray(i, dtype='float32') for i in enc_mask]
        dec_data = [np.asarray(i, dtype='float32') for i in dec_data]
        dec_mask  = [np.asarray(i, dtype='float32') for i in dec_mask]
        label = [np.asarray(i, dtype='float32') for i in label]
        return enc_data, enc_mask, dec_data, dec_mask, label