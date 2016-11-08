# pylint: disable=C0111,too-many-arguments,too-many-instance-attributes,too-many-locals,redefined-outer-name,fixme
# pylint: disable=superfluous-parens, no-member, invalid-name

import numpy as np
import mxnet as mx

from y_lstm import lstm_unroll
from y_bucket_io import BucketSentenceIter

if __name__ == '__main__':
# ---------------- Preprocess ---------------- 
	vocab_size = 10000

# ---------------- Params Defination ----------------
  ## network params defination
	num_lstm_layer = 2
	num_hidden = 256
	num_embed = 128
	input_size = vocab_size
	num_label = input_size

  ## training params defination
	contexts = [mx.context.cpu(i) for i in range(1)]
	batch_size = 32
	num_epoch = 25
	learning_rate = 0.01
	momentum = 0.0

  ## lstm specific params defination
	buckets = [10, 20, 30, 40, 50, 60]
	init_c = [('l%d_init_c'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
	init_h = [('l%d_init_h'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
	init_states = init_h + init_c

 ## testing params defination
	dummy_data = False # dummy data is used to test speed without IO

#---------------- DataIterator Defination ----------------
	data_train = BucketSentenceIter(
						path = "./data/ptb.train.txt",
						vocab_size = vocab_size,
						buckets = buckets,
						batch_size = batch_size,
						init_states = init_states
				)
	data_val = BucketSentenceIter(
						path = "./data/ptb.valid.txt",
						vocab_size = vocab_size,
						buckets = buckets,
						batch_size = batch_size,
						init_states = init_states
				)

	if dummy_data:
			data_train = DummyIter(data_train)
			data_val = DummyIter(data_val)

# ---------------- Training Defination ----------------
	def Perplexity(labels, preds):
		loss = 0.
		for i in range(preds.shape[0]):
			loss += -np.log(max(1e-10, preds[i][int(labels[i])]))
		return np.exp(loss / labels.size)

	def sym_gen(seq_len):
			return lstm_unroll(
						num_lstm_layer = num_lstm_layer,
						seq_len = seq_len,
						input_size = vocab_size,
						num_hidden = num_hidden,
						num_embed = num_embed,
						num_label = vocab_size
					)

	if len(buckets) == 1:
			# only 1 bucket, disable bucketing
			symbol = sym_gen(buckets[0])
	else:
			symbol = sym_gen

	model = mx.model.FeedForward(
								ctx=contexts,
								symbol=symbol,
								num_epoch=num_epoch,
								learning_rate=learning_rate,
								momentum=momentum,
								wd=0.00001,
								initializer=mx.init.Xavier(factor_type="in", magnitude=2.34)
	)

	import logging
	head = '%(asctime)-15s %(message)s'
	logging.basicConfig(level=logging.DEBUG, format=head)

	model.fit(
			X=data_train,
			eval_data=data_val,
			eval_metric = mx.metric.np(Perplexity),
			batch_end_callback=mx.callback.Speedometer(batch_size, 50)
	)

