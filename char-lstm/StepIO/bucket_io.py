# pylint: disable=C0111,too-many-arguments,too-many-instance-attributes,too-many-locals,redefined-outer-name,fixme
# pylint: disable=superfluous-parens, no-member, invalid-name
import collections
import numpy as np
import mxnet as mx

# The interface of a data iter that works for bucketing
#
# DataIter
#   - default_bucket_key: the bucket key for the default symbol.
#
# DataBatch
#   - provide_data: same as DataIter, but specific to this batch
#   - provide_label: same as DataIter, but specific to this batch
#   - bucket_key: the key for the bucket that should be used for this batch

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

class BucketSentenceIter(mx.io.DataIter):
	
	def __init__(self, path, vocab_size,
				buckets, batch_size, init_states,
				data_name='data', mask_name = 'mask', label_name='label',
				seperate_char=' <eos> '):
		super(BucketSentenceIter, self).__init__()
		# Initialization
		self.data_name = data_name
		self.mask_name = mask_name
		self.label_name = label_name
		self.buckets = sorted(buckets)
		self.default_bucket_key = max(buckets)
		self.data = [[] for _ in buckets]
		self.mask = [[] for _ in buckets]
		self.pad_label = 0
		# Preprocess data
		self.path = path
		self.vocab_size = vocab_size
		self.content =  self.read_content(path)
		self.build_vocab(path)
		sentences = self.content.split(seperate_char)
		
		discard_cnt = 0

		for sentence in sentences:
			sentence = self.text2id(sentence)
			if len(sentence) == 1:
				continue
			bkt_idx = self.find_bucket(len(sentence))
			if bkt_idx == -1:
				discard_cnt += 1
				continue
			d, m = self.make_data(sentence, self.buckets[bkt_idx])
			self.data[bkt_idx].append(d)
			self.mask[bkt_idx].append(m)

		# convert data into ndarrays for better speed during training
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
		self.provide_data = [('%s/%d' % (self.data_name, t), (self.batch_size,)) for t in range(self.default_bucket_key)] +\
		[('%s/%d' % (self.mask_name, k), (self.batch_size,)) for k in range(self.default_bucket_key)] + init_states
		self.provide_label = [('%s/%d' % (self.label_name, t), (self.batch_size,))
								for t in range(self.default_bucket_key)]

		self.reset()

	def read_content(self, path):
		with open(path) as ins:
			content = ins.read()
			content = content.replace('\n', ' <eos> ').replace('. ', ' <eos> ')
			return content
	
	'''
	take top k and abondon others as unknown
	0 is left for padding 
	last is left for unknown
	'''
	def build_vocab(self, path):
		cnt = collections.Counter(self.content.split(' '))
		keys = cnt.most_common(self.vocab_size - 2)
		self.dic = {'PAD' : 0}
		self.reverse_dic = {0 : 'PAD', self.vocab_size - 1 : "<UNK>"} # is useful for inference from RNN
		for i in range(len(keys)):
			k = keys[i][0]
			v = i + 1
			self.dic[k] = v
			self.reverse_dic[v] = k
		print("Total tokens: %d, keep %d" % (len(cnt), self.vocab_size))
	
	def text2id(self, sentence):
		sentence += "<eos>"
		words = sentence.split(' ')
		idx = [0] * len(words)
		for i in range(len(words)):
			if words[i] in self.dic:
				idx[i] = self.dic[words[i]]
			else:
				idx[i] = self.vocab_size - 1
		return idx

	def find_bucket(self, val):
	# lazy to use O(n) way
		for i, bkt in enumerate(self.buckets):
			if bkt > val:
				return i
		return -1

	def make_data(self, sentence, bucket):
		# pad at the begining of the sequence
		mask = [1] * bucket
		data = [self.pad_label] * bucket
		pad = bucket - len(sentence)
		data[pad:] = sentence
		mask[:pad] = [0 for i in range(pad)]
		return data, mask

	def make_data_iter_plan(self):
		"make a random data iteration plan"
		# truncate each bucket into multiple of batch-size
		bucket_n_batches = []
		for i in range(len(self.data)):
			bucket_n_batches.append(len(self.data[i]) / self.batch_size)
			self.data[i] = self.data[i][:int(bucket_n_batches[i]*self.batch_size)]

		bucket_plan = np.hstack([np.zeros(n, int)+i for i, n in enumerate(bucket_n_batches)])
		np.random.shuffle(bucket_plan)

		bucket_idx_all = [np.random.permutation(len(x)) for x in self.data]

		self.bucket_plan = bucket_plan
		self.bucket_idx_all = bucket_idx_all
		self.bucket_curr_idx = [0 for x in self.data]

		self.data_buffer = []
		self.mask_buffer = []
		self.label_buffer = []

		for i_bucket in range(len(self.data)):
			data = np.zeros((self.batch_size, self.buckets[i_bucket]))
			mask = np.zeros((self.batch_size, self.buckets[i_bucket]))
			label = np.zeros((self.batch_size, self.buckets[i_bucket]))
			self.data_buffer.append(data)
			self.mask_buffer.append(mask)
			self.label_buffer.append(label)

	def __iter__(self):
		init_state_names = [x[0] for x in self.init_states]

		for i_bucket in self.bucket_plan:
			data = self.data_buffer[i_bucket]
			mask = self.mask_buffer[i_bucket]
			label = self.label_buffer[i_bucket]

			i_idx = self.bucket_curr_idx[i_bucket]
			idx = self.bucket_idx_all[i_bucket][i_idx:i_idx+self.batch_size]
			self.bucket_curr_idx[i_bucket] += self.batch_size
			
			data[:] = self.data[i_bucket][idx]
			label[:, :-1] = data[:, 1:]
			label[:, -1] = 0
			mask[:] = self.mask[i_bucket][idx]

			data_all = [mx.nd.array(data[:, t]) for t in range(self.buckets[i_bucket])] +\
			[mx.nd.array(mask[:, t]) for t in range(self.buckets[i_bucket])] + self.init_state_arrays
			label_all = [mx.nd.array(label[:, t]) for t in range(self.buckets[i_bucket])]
			data_names = ['%s/%d' % (self.data_name, t) for t in range(self.buckets[i_bucket])] +\
			['%s/%d' % (self.mask_name, t) for t in range(self.buckets[i_bucket])] + init_state_names
			label_names = ['%s/%d' % (self.label_name, t) for t in range(self.buckets[i_bucket])]

			data_batch = SimpleBatch(data_names, data_all, label_names, label_all,
							self.buckets[i_bucket])
			yield data_batch

	def reset(self):
		self.bucket_curr_idx = [0 for x in self.data]
