import logging
import codecs
import re
from collections import namedtuple

import mxnet as mx
import numpy as np
from sklearn.cluster import KMeans

EncDecBucketKey = namedtuple('EncDecBucketKey', ['enc_len', 'dec_len'])

class EncoderDecoderBatch(object):
    def __init__(self, data_names, data, label_names, label, bucket_key):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names
        self.bucket_key = bucket_key

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]

class DummyIter(mx.io.DataIter):
    "A dummy iterator that always return the same batch, used for speed testing"
    def __init__(self, real_iter):
        super(DummyIter, self).__init__()
        self.real_iter = real_iter
        self.provide_data = real_iter.provide_data
        self.provide_label = real_iter.provide_label
        self.batch_size = real_iter.batch_size

        for batch in real_iter:
            self.the_batch = batch
            break

    def __iter__(self):
        return self

    def next(self):
        return self.the_batch

def read_dict(path):
    word2idx = {'<PAD>' : 0, '<EOS>' : 1, '<UNK>' : 2}
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
    with codecs.open(path, 'r', encoding='utf-8', errors='ignore') as fid:
        for line in fid:
            line = line.strip()
            line_list = line.split('\t=>\t')
            length = len(line_list)
            for i in xrange(1, length):
                enc_list = line_list[0].strip().split()
                dec_list = line_list[i].strip().split()
                enc = [enc_word2idx.get(word) if enc_word2idx.get(word) != None else enc_word2idx.get('<UNK>') for word in enc_list]
                dec = [dec_word2idx.get(word) if dec_word2idx.get(word) != None else  dec_word2idx.get('<UNK>') for word in dec_list]
                enc_data.append(enc)
                dec_data.append(dec)
    return enc_data, dec_data

class EncoderDecoderIter(mx.io.DataIter):
    def __init__(self, enc_word2idx, dec_word2idx, filename, #init_states,
                batch_size = 20, num_buckets = 5, DEBUG = False):
        
        super(EncoderDecoderIter, self).__init__()
        # initilization
        self.enc_word2idx = enc_word2idx
        self.dec_word2idx = dec_word2idx
        self.pad = self.enc_word2idx.get('<PAD>')
        self.eos = self.enc_word2idx.get('<EOS>')
        self.enc_data, self.dec_data = get_enc_dec_text_id(filename, enc_word2idx, dec_word2idx)
        print 'Text2digit Preprocess Example:'
        for i in range(4):
            print self.enc_data[i], self.dec_data[i]
        self.data_len = len(self.enc_data)
        self.DEBUG = DEBUG
        self.enc_data_name = 'enc_data'
        self.dec_data_name = 'dec_data'
        self.enc_mask_name = 'enc_mask'
        self.dec_mask_name = 'dec_mask'
        self.label_name = 'label'
        self.batch_size = batch_size

        #self.init_states = init_states
        #self.init_state_arrays = [mx.nd.zeros(x[1]) for x in self.init_states]
        
        # Automatically generate buckets
        self.num_buckets = num_buckets
        self.generate_buckets()
        
        # generate the data , mask, label numpy array
        self.make_data_array()


        # make a random data iteration plan
        self.make_data_iter_plan()
        self.reset()

    def make_data_array(self):
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
            #if DEBUG:
                #print i,  ed, em, dd, dm, l
                #print "================="
        self.enc_data = [np.zeros((len(x), self.buckets[i].enc_len)) for i, x in enumerate(enc_data)]
        self.enc_mask  = [np.zeros((len(x), self.buckets[i].enc_len)) for i, x in enumerate(enc_data)]
        self.dec_data = [np.zeros((len(x), self.buckets[i].dec_len)) for i, x in enumerate(enc_data)]
        self.dec_mask  = [np.zeros((len(x), self.buckets[i].dec_len)) for i, x in enumerate(enc_data)]
        self.label = [np.zeros((len(x), self.buckets[i].dec_len)) for i, x in enumerate(enc_data)]
        for bkt_idx in xrange(len(self.buckets)):
            for j in xrange(len(enc_data[bkt_idx])):
                self.enc_data[bkt_idx][j, :] = enc_data[bkt_idx][j]
                self.enc_mask[bkt_idx][j, :] = enc_mask[bkt_idx][j]
                self.dec_data[bkt_idx][j, :] = dec_data[bkt_idx][j]
                self.dec_mask[bkt_idx][j, :] = dec_mask[bkt_idx][j]
                self.label[bkt_idx][j, :] = label[bkt_idx][j]


    @property
    def provide_data(self):
        p_data = [('enc_data' , (self.batch_size, self.default_bucket_key.enc_len)),
                  ('enc_mask' , (self.batch_size, self.default_bucket_key.enc_len)),
                  ('dec_data' , (self.batch_size, self.default_bucket_key.dec_len)),
                  ('dec_mask' , (self.batch_size, self.default_bucket_key.dec_len)),] #+ self.init_states
        return p_data

    @property
    def provide_label(self):
        p_label = [('label', (self.batch_size, self.default_bucket_key.dec_len))]
        return p_label

    def __iter__(self):
        #init_state_names = [x[0] for x in self.init_states]

        for i_bucket in self.bucket_plan:
            enc_data = self.enc_data_buffer[i_bucket]
            enc_mask = self.enc_mask_buffer[i_bucket]
            dec_data = self.dec_data_buffer[i_bucket]
            dec_mask = self.dec_mask_buffer[i_bucket]
            label = self.label_buffer[i_bucket]

            i_idx = self.bucket_curr_idx[i_bucket]
            idx = self.bucket_idx_all[i_bucket][i_idx:i_idx+self.batch_size]
            self.bucket_curr_idx[i_bucket] += self.batch_size
            
            enc_data[:] = self.enc_data[i_bucket][idx]
            enc_mask[:] = self.enc_mask[i_bucket][idx]
            dec_data[:] = self.dec_data[i_bucket][idx]
            dec_mask[:] = self.dec_mask[i_bucket][idx]      
            label[:] = self.label[i_bucket][idx]

            data_all = [mx.nd.array(enc_data), mx.nd.array(enc_mask),mx.nd.array(dec_data), mx.nd.array(dec_mask) ]  #+ self.init_state_arrays
            label_all = [mx.nd.array(label)]
            
            data_names = ['enc_data', 'enc_mask', 'dec_data', 'dec_mask'] #+ init_state_names
            label_names = ['label']

            data_batch = EncoderDecoderBatch(data_names, data_all, label_names, label_all,
                            self.buckets[i_bucket])
            yield data_batch

    def reset(self):
        self.bucket_curr_idx = [0 for x in self.enc_data]
        np.random.shuffle(self.bucket_plan)
        for bucket in self.bucket_idx_all:
            np.random.shuffle(bucket)

    def make_data_iter_plan(self):
        "make a random data iteration plan"
        # truncate each bucket into multiple of batch-size
        bucket_n_batches = []
        for i in range(len(self.enc_data)):
            bucket_n_batches.append(len(self.enc_data[i]) / self.batch_size)
            #self.enc_data[i] = self.enc_data[i][:int(bucket_n_batches[i]*self.batch_size)]
        bucket_plan = np.hstack([np.zeros(n, int)+i for i, n in enumerate(bucket_n_batches)])
        np.random.shuffle(bucket_plan)
        bucket_idx_all = [np.random.permutation(len(x)) for x in self.enc_data]
        #bucket_idx_all = [ np.array(range(len(x))) for x in self.enc_data]

        self.bucket_plan = bucket_plan
        self.bucket_idx_all = bucket_idx_all
        self.bucket_curr_idx = [0 for x in self.enc_data]
        self.enc_data_buffer = []
        self.enc_mask_buffer = []
        self.dec_data_buffer = []
        self.dec_mask_buffer = []
        self.label_buffer = []

        for i_bucket in range(len(self.enc_data)):
            enc_data = np.zeros((self.batch_size, self.buckets[i_bucket].enc_len))
            enc_mask = np.zeros((self.batch_size, self.buckets[i_bucket].enc_len))
            dec_data = np.zeros((self.batch_size, self.buckets[i_bucket].dec_len))
            dec_mask = np.zeros((self.batch_size, self.buckets[i_bucket].dec_len))
            label = np.zeros((self.batch_size, self.buckets[i_bucket].dec_len))
            self.enc_data_buffer.append(enc_data)
            self.enc_mask_buffer.append(enc_mask)
            self.dec_data_buffer.append(dec_data)
            self.dec_mask_buffer.append(dec_mask)
            self.label_buffer.append(label)


    def make_data_line(self, i, bucket):
        data = self.enc_data[i]
        label = self.dec_data[i]
        enc_len = bucket.enc_len
        dec_len = bucket.dec_len
        
        ed = np.full(enc_len, self.pad, dtype = float)
        dd = np.full(dec_len, self.pad, dtype = float)
        em = np.zeros(enc_len, dtype = float)
        dm = np.zeros(dec_len, dtype = float)
        l  = np.full(dec_len, self.pad, dtype = float)
        
        ed[enc_len-len(data):enc_len] = data
        em[enc_len-len(data):enc_len] = 1.0
        dd[0] = self.eos 
        dd[1:len(label)+1] = label 
        dm[0:len(label)+1] = 1.0
        l[0:len(label)] = label
        l[len(label)] = self.eos

        return ed, em, dd, dm, l

    def generate_buckets(self):
        enc_dec_data = []
        for i in xrange(self.data_len):
            enc_len = len(self.enc_data[i])  
            dec_len = len(self.dec_data[i]) + 1 # plus one because of the <EOS>
            enc_dec_data.append((enc_len, dec_len))
        enc_dec_data = np.array(enc_dec_data)
        kmeans = KMeans(n_clusters = self.num_buckets, random_state = 1) # use clustering to decide the buckets
        self.assignments = kmeans.fit_predict(enc_dec_data) # get the assignments
        # get the max of every cluster
        self.buckets = np.array([np.max( enc_dec_data[self.assignments==i], axis=0 ) for i in range(self.num_buckets)])
        # get # of sequences in each bucket... then assign the batch size as the minimum(minimum(bucketsize), batchsize)
        buckets_count = np.array( [ enc_dec_data[self.assignments==i].shape[0] for i in range(self.num_buckets) ] )
        
        if self.DEBUG:
            print 'self.buckets: ', self.buckets
            print 'self.buckets_count: ', self.buckets_count
            print 'assignments: ', self.assignments
        buckets = []
        for i in xrange(self.num_buckets):
            buckets.append(EncDecBucketKey(enc_len = self.buckets[i][0], dec_len = self.buckets[i][1]))
        self.buckets = buckets
        enc_len, dec_len = np.max(self.buckets, axis=0)
        self.default_bucket_key = EncDecBucketKey(enc_len = enc_len, dec_len = dec_len)
