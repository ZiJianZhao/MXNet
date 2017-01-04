import mxnet as mx
import numpy as np

from sklearn.cluster import KMeans

class SimpleBatch(object):
	def __init__(self, data_names, data,
			label_names, label, bucket_key):
		
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

DEBUG = False

class SequenceIter(mx.io.DataIter):
	def __init__(self, data, label, 
			pad, init_states, 
			batch_size = 32, num_buckets = 5):
		# Initialization
		super(SequenceIter, self).__init__() 
		
		self.data = data
		self.label = label 
		self.data_num = len(data)

		self.data_name = ['data', 'mask']
		self.label_name = 'label'

		self.batch_size = batch_size
		self.pad = pad 
		# Automatically generate buckets
		self.num_buckets = num_buckets
		self.buckets, self.buckets_count, assignments = self.generate_buckets()
		if DEBUG:
			print 'self.buckets: ', self.buckets
			print 'self.buckets_count: ', self.buckets_count
			
		buckets = []
		for i in xrange(num_buckets):
			buckets.append(self.buckets[i][0])
		self.buckets = buckets

		self.default_bucket_key = max(self.buckets)
		
		# generate the data , mask, label numpy array
		data = [[] for _ in self.buckets]
		mask = [[] for _ in self.buckets]
		label  = [[] for _ in self.buckets]
		for i in xrange(self.data_num):
			bkt_idx = assignments[i]
			i_data, i_mask, i_label = self.make_data(i, self.buckets[bkt_idx])
			data[bkt_idx].append(i_data)
			mask[bkt_idx].append(i_mask)
			label[bkt_idx].append(i_label)

		self.data = [np.zeros((len(x), self.buckets[i])) for i, x in enumerate(data)]
		self.mask  = [np.zeros((len(x), self.buckets[i])) for i, x in enumerate(data)]
		self.label = [np.zeros((len(x), self.buckets[i])) for i, x in enumerate(label)]

		for bkt_idx in xrange(len(self.buckets)):
			for j in xrange(len(data[bkt_idx])):
				self.data[bkt_idx][j, :] = data[bkt_idx][j]
				self.mask[bkt_idx][j, :] = mask[bkt_idx][j]
				self.label[bkt_idx][j, :] = label[bkt_idx][j]

		# make a random data iteration plan
		self.make_data_iter_plan()
		self.init_states = init_states
		self.init_state_arrays = [mx.nd.zeros(x[1]) for x in self.init_states]
		self.provide_data = [('data' , (self.batch_size, self.default_bucket_key)),
					('mask' , (self.batch_size, self.default_bucket_key)),] + self.init_states
		self.provide_label = [('label', (self.batch_size, self.default_bucket_key))]
		self.reset()

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
			mask[:] = self.mask[i_bucket][idx]
			label[:] = self.label[i_bucket][idx]

			data_all = [mx.nd.array(data), mx.nd.array(mask)]   + self.init_state_arrays
			label_all = [mx.nd.array(label)]
			data_names = ['data', 'mask'] + init_state_names
			label_names = ['label']

			data_batch = SimpleBatch(data_names, data_all, label_names, label_all,
							self.buckets[i_bucket])
			yield data_batch

	def reset(self):
		self.bucket_curr_idx = [0 for x in self.data]	

	def make_data_iter_plan(self):
		"make a random data iteration plan"
		# truncate each bucket into multiple of batch-size
		bucket_n_batches = []
		for i in xrange(len(self.data)):
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

	def make_data(self, i, bucket):
		data = self.data[i]
		label = self.label[i]
		
		i_data = np.full(bucket, self.pad, dtype = float)
		i_label  = np.full(bucket, self.pad, dtype = float)
		i_mask = np.zeros(bucket, dtype = float)
		i_data[:len(data)] = data
		i_label[:len(label)] = label
		i_mask[:len(data)] = 1
		return i_data, i_mask, i_label

	def generate_buckets(self):
		sequence_length = []
		for i in xrange(self.data_num):
			sequence_length.append(len(self.data[i]))

		sequence_length = np.array(sequence_length).reshape(-1,1)
		kmeans = KMeans(n_clusters = self.num_buckets, random_state = 1) # use clustering to decide the buckets
		assignments = kmeans.fit_predict(sequence_length) # get the assignments

		# get the max of every cluster
		buckets = np.array([np.max( sequence_length[assignments==i], axis=0 ) for i in range(self.num_buckets)])

		# get # of sequences in each bucket... then assign the batch size as the minimum(minimum(bucketsize), batchsize)
		buckets_count = np.array([sequence_length[assignments==i].shape[0] for i in range(self.num_buckets) ] )

		return buckets, buckets_count, assignments