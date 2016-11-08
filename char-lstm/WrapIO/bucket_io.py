import collections
import mxnet as mx
import numpy as np

# The interface of a data iter that works for bucketing
#
# DataIter
#   - default_bucket_key: the bucket key for the default symbol.
#
# DataBatch
#   - provide_data: same as DataIter, but specific to this batch
#   - provide_label: same as DataIter, but specific to this batch
#   - bucket_key: the key for the bucket that should be used for this batch



# simple batch is used for module to get data name, label name, data, label and bucket key
class SimpleBatch(object):
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


class BucketSentenceIter(mx.io.DataIter):
    def __init__(self, path, buckets, vocab_size, batch_size, init_states):
        super(BucketSentenceIter, self).__init__()
        self.path = path
        self.buckets = sorted(buckets)
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        # init
        self.data_name = ['data', 'mask']
        self.label_name = 'softmax_label'
        self._preprocess()
        self._build_vocab()
        sentences = self.content.split('<eos>')
        self.data = [[] for _ in self.buckets]
        self.mask = [[] for _ in self.buckets]
        # pre-allocate with the largest bucket for better memory sharing
        self.default_bucket_key = max(buckets)

        discard_cnt = 0

        for sentence in sentences:
            sentence= self._text2id(sentence)
            bkt_idx = self._find_bucket(len(sentence))
            if bkt_idx == -1:
                discard_cnt += 1
                continue
            d, m = self._make_data(sentence, self.buckets[bkt_idx])
            self.data[bkt_idx].append(d)
            self.mask[bkt_idx].append(m)


        # convert data into ndarrays for better speed during training
        # default the pad is 0
        data = [np.zeros((len(x), buckets[i])) for i, x in enumerate(self.data)]
        mask = [np.zeros((len(x), buckets[i])) for i, x in enumerate(self.data)]
        for i_bucket in range(len(self.buckets)):
            for j in range(len(self.data[i_bucket])):
                data[i_bucket][j, :] = self.data[i_bucket][j]
                mask[i_bucket][j, :] = self.mask[i_bucket][j]

        self.data = data
        self.mask = mask

        # Get the size of each bucket, so that we could sample
        # uniformly from the bucket
        bucket_sizes = [len(x) for x in self.data]

        print("Summary of dataset ==================")
        print("Discard instance: %3d" % discard_cnt)
        for bkt, size in zip(buckets, bucket_sizes):
            print("bucket of len %3d : %d samples" % (bkt, size))

        self.batch_size = batch_size
        self.make_data_iter_plan()

        self.init_states = init_states
        self.init_state_arrays = [mx.nd.zeros(x[1]) for x in init_states]

        self.provide_data = [('data', (batch_size, self.default_bucket_key)),
                             ('mask', (batch_size, self.default_bucket_key))] + init_states
        self.provide_label = [('softmax_label', (self.batch_size, self.default_bucket_key))]
        #self.provide_label = [(symbol(p.prefix, "softmax_label_$t"), (p.batch_size,)) for t = 1:p.seq_len]
        self.reset()

    def _preprocess(self):
        self.content = open(self.path).read()

    def _find_bucket(self, val):
        # lazy to use O(n) way
        for i, bkt in enumerate(self.buckets):
            if bkt > val:
                return i
        return -1

    def _make_data(self, sentence, bucket):
        # pad at the begining of the sequence
        mask = [1] * bucket
        data = [0] * bucket
        pad = bucket - len(sentence)
        data[pad:] = sentence
        mask[:pad] = [0 for i in range(pad)]
        return data, mask

    def _gen_bucket(self, sentence):
        # you can think about how to generate bucket candidtes in heuristic way
        # here we directly use manual defined buckets
        return self.buckets


    def _build_vocab(self):
        cnt = collections.Counter(list(self.content))
        # take top k and abandon others as unknown
        # 0 is left for padding
        # last is left for unknown
        
        # reserve all the words
        #self.vocab_size = len(cnt.keys()) + 2

        keys = cnt.most_common(self.vocab_size - 1)
        self.dic = {'PAD' : 0}
        self.reverse_dic = {0 : 'PAD'} # is useful for inference from RNN
        for i in range(len(keys)):
            k = keys[i][0]
            v = i + 1
            self.dic[k] = v
            self.reverse_dic[v] = k
        print("Total tokens: %d, keep %d" % (len(cnt), self.vocab_size))


    def _text2id(self, sentence):   
        sentence += " <eos>"
        words = sentence.split(' ')
        idx = [0] * len(words)
        for i in range(len(words)):
            if words[i] in self.dic:
                idx[i] = self.dic[words[i]]
            else:
                idx[i] = self.vocab_size - 1
        return idx
        


    def next(self):
        init_state_names = [x[0] for x in self.init_states]
        
        for i_bucket in self.bucket_plan:
            data = self.data_buffer[i_bucket]
            i_idx = self.bucket_curr_idx[i_bucket]
            idx = self.bucket_idx_all[i_bucket][i_idx:i_idx+self.batch_size]
            self.bucket_curr_idx[i_bucket] += self.batch_size
            init_state_names = [x[0] for x in self.init_states]
            data[:] = self.data[i_bucket][idx]

            for sentence in data:
                assert len(sentence) == self.buckets[i_bucket]
                
            label = self.label_buffer[i_bucket]
            label[:, :-1] = data[:, 1:]
            label[:, -1] = 0

            mask = self.mask_buffer[i_bucket]
            mask[:] = self.mask[i_bucket][idx]

            data_all = [mx.nd.array(data), mx.nd.array(mask)] + self.init_state_arrays
            label_all = [mx.nd.array(label)]
            data_names = ['data', 'mask'] + init_state_names
            label_names = ['softmax_label']

            data_batch = SimpleBatch(data_names, data_all, label_names, label_all,
                                         self.buckets[i_bucket])
            yield data_batch

    __iter__ = next

    def reset(self):
        self.bucket_curr_idx = [0 for x in self.data]
        

    def make_data_iter_plan(self):
        "make a random data iteration plan"
        # truncate each bucket into multiple of batch-size
        bucket_n_batches = []
        for i in range(len(self.data)):
            bucket_n_batches.append(int(len(self.data[i]) / self.batch_size))
            self.data[i] = self.data[i][:int(bucket_n_batches[i]*self.batch_size)]
        bucket_plan = np.hstack([np.zeros(n, int)+i for i, n in enumerate(bucket_n_batches)])
        np.random.shuffle(bucket_plan)

        bucket_idx_all = [np.random.permutation(len(x)) for x in self.data]

        self.bucket_plan = bucket_plan
        self.bucket_idx_all = bucket_idx_all
        self.bucket_curr_idx = [0 for x in self.data]

        self.data_buffer = []
        self.label_buffer = []
        self.mask_buffer = []

        for i_bucket in range(len(self.data)):
            data = np.zeros((self.batch_size, self.buckets[i_bucket]))
            label = np.zeros((self.batch_size, self.buckets[i_bucket]))
            mask = np.zeros((self.batch_size, self.buckets[i_bucket]))
            self.data_buffer.append(data)
            self.label_buffer.append(label)
            self.mask_buffer.append(mask)


    def reset_states(states_data=None):
        if states_data == None:
            for arr in self.init_state_arrays:
                arr[:] = 0
        else:
            assert len(states_data) == len(self.init_state_arrays)
            for i in range(len(states_data)):
                states_data[i].copyto(self.init_state_arrays[i])