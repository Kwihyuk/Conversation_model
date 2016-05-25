import random
import sys
import os
import time
import math




import numpy as np
import tensorflow as tf
import data_utils
import seq2seq_model
# i don't know why below import needed

from six.moves import xrange
tf.app.flags.DEFINE_float("learning_rate", 		0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99, "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 		5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size",      		64, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("cell_size",             	1024, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 		1, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("source_target_vocab_size",	40000, "vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", 			"../data", "Data directory")
tf.app.flags.DEFINE_string("train_dir", 		"./train_model", "Training directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 	0, "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 	200, "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode",			False, "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", 		False, "Run a self-test if this is set to True.")

FLAGS = tf.app.flags.FLAGS




_buckets = [ (5,10), (10,15), (20,25), (30,40), (40,50), (50,60)]


def read_data(source_path,target_path, max_size=None):
	""" Read data from source and targt files and put into buckets.

	Args:

	it must be aligned with the source file : n-th line contains the desired output
	for n-th line from the source_path

	max_size : maximum number of lines to read, all other will be ignored;
		   if 0 or None, data files will be read completely ( no limit )

	Returns:
		data_set : a list of length len(_buckets);
			   data_set[n] contains a list of (source, target) paris read from the
			   provided data filed that fit into the n-th bucket
			   i.e., such that len(source) < _buckets[n][0] and
			                   len(target) < _buckets[n][1]
			         source and target are lists of token-ids.
        """

	print 'extracting data'
	data_set = [  []  for _ in _buckets ]
	with tf.gfile.GFile(source_path,mode="r") as A_file:
		with tf.gfile.GFile(target_path,mode="r") as Q_file:
			source, target = A_file.readline(), Q_file.readline()
			counter = 0
			while source and target and ( not max_size or counter < max_size):
				counter += 1
				if counter % 10000 == 0:
					print(" reading data line %d" % ( counter ))
					sys.stdout.flush()
				source_ids = [int(x)+3 for x in source.split()]
				target_ids = [int(x)+3 for x in target.split()]
				target_ids.append(data_utils.EOS_ID)
				for bucket_id, (source_size, target_size) in enumerate(_buckets):
					if len(source_ids) < source_size and len(target_ids) < target_size:
						data_set[bucket_id].append([source_ids, target_ids])
						break
				source, target = A_file.readline(),Q_file.readline()
	return data_set









def create_model(session, forward_only):
	""" Create conversation model and initialize or load parameter in session."""

	"""Seq2SeqModel : Args
		*source_target_vocab_size: size of the source/target vocabulary.
		*buckets: a list of pairs (I, O), where I specifies maximum input length
			 that will be processed in that bucket, and O specifies maximum output
			 length. Training instances that have inputs longer than I or outputs
			 longer than O will be pushed to the next bucket and padded accordingly.
			 We assume that the list is sorted, e.g., [(2, 4), (8, 16)].
		*size: number of units in each layer of the model.
		*num_layers: number of layers in the model.
		*max_gradient_norm: gradients will be clipped to maximally this norm.
		*batch_size: the size of the batches used during training;
			    the model construction is independent of batch_size, so it can be
			    changed after initialization if this is convenient, e.g., for decoding.
		*learning_rate: learning rate to start with.
		*learning_rate_decay_factor: decay learning rate by this much when needed.
		*forward_only: if set, we do not construct the backward pass in the model.
	"""
	model = seq2seq_model.Seq2SeqModel( FLAGS.source_target_vocab_size, _buckets, FLAGS.cell_size,
					    FLAGS.num_layers, FLAGS.max_gradient_norm,
					    FLAGS.batch_size, FLAGS.learning_rate,
					    FLAGS.learning_rate_decay_factor,
					    forward_only = forward_only)

	ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
	if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
		print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
		model.saver.restore(session, ckpt.model_checkpoint_path)
	else:
		print("Created model with fresh parameters.")
		session.run(tf.initialize_all_variables())
	return model



def train():
   """ Train Conversation Model using Movie conversation corpus """


   print ("Preparing move conversaion data in %s" %FLAGS.data_dir)


   with tf.Session() as sess:
      # Create model
      print ( "Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.cell_size) )
      model = create_model(sess, False)


      # Read data into buckets and comput their sizes.
      print ("Reading Testing and Training data" )
      train_set = read_data('../data/Train_Answer_token_ids.txt','../data/Train_Question_token_ids.txt')
      test_set = read_data('../data/Test_Answer_token_ids.txt','../data/Test_Question_token_ids.txt')
      print train_set[0][2]
      train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
      train_total_size = float(sum(train_bucket_sizes))

      print train_bucket_sizes
      print "%d " % (train_total_size)


      # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
      # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
      # the size if i-th training bucket, as used later.

      train_buckets_scale = [sum(train_bucket_sizes[:i+1]) / train_total_size for i in xrange(len(train_bucket_sizes))]

      print train_buckets_scale


      step_time, loss = 0.0, 0.0
      current_step = 0
      previous_losses = []
      while True:
		  #Choose a bucket according to data distribution. We pick a random number in [0,1] and use the corresponding interval in train_buckets_scale.
		  random_number_01 = np.random.random_sample()
		  bucket_id = min( [i for i in xrange(len(train_buckets_scale))
                         if train_buckets_scale[i] > random_number_01] )
		  print bucket_id
		  start_time = time.time()
		  encoder_inputs, decoder_inputs, target_weights = model.get_batch(train_set, bucket_id)

		  # encoder_inputs = [# of length][# of batch_size]
		  print np.shape(encoder_inputs)

		  _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, False)

















def main(unused_args):
	train()
#	data_set = read_data()
#	print data_set[0][0]
#
#	with tf.Session() as sess:
#		print ("Test")
#	model = create_model(sess,False)
#	sess.run(tf.initialize_all_variables())

if __name__ == "__main__":
	tf.app.run()
