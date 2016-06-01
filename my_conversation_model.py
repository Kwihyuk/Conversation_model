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
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.999, "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 		5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size",      		64, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("cell_size",             	512, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 		3, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("source_target_vocab_size",	40874, "vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", 			"../data", "Data directory")
tf.app.flags.DEFINE_string("train_dir", 		"./train_model", "Training directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 	0, "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 	200, "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("self_test", 		False, "Run a self-test if this is set to True.")

FLAGS = tf.app.flags.FLAGS




#_buckets = [ (5,10), (10,15), (20,25), (30,40), (40,50), (50,60)]

_buckets = [ (5,5), (10,10), (15,15), (20,20), (30,30), (40,40), (50,50)]

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
				if counter % 5000 == 0:
					print(" reading data line %d" % ( counter ))
					sys.stdout.flush()
				source_ids = [int(x) for x in source.split()]
				target_ids = [int(x) for x in target.split()]
				target_ids.append(data_utils.EOS_ID)
				for bucket_id, (source_size, target_size) in enumerate(_buckets):
					if len(source_ids) < source_size and len(target_ids) < target_size:
						data_set[bucket_id].append([source_ids, target_ids])
						break
				source, target = A_file.readline(),Q_file.readline()
	print (" reading data line %d ( total data ) done" % (counter) )
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
      #print train_set[0][2]
      train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
      train_total_size = float(sum(train_bucket_sizes))

      #print train_bucket_sizes
      #print "%d " % (train_total_size)


      # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
      # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
      # the size if i-th training bucket, as used later.

      train_buckets_scale = [sum(train_bucket_sizes[:i+1]) / train_total_size for i in xrange(len(train_bucket_sizes))]

      #print train_buckets_scale


      step_time, loss = 0.0, 0.0
      current_step = 0
      previous_losses = []
      while True:
		  #Choose a bucket according to data distribution. We pick a random number in [0,1] and use the corresponding interval in train_buckets_scale.
		  random_number_01 = np.random.random_sample()
		  bucket_id = min( [i for i in xrange(len(train_buckets_scale))
                         if train_buckets_scale[i] > random_number_01] )
		  #print bucket_id
		  start_time = time.time()
		  encoder_inputs, decoder_inputs, target_weights = model.get_batch(train_set, bucket_id)

		  # encoder_inputs = [# of length][# of batch_size]
		  #print np.shape(encoder_inputs)

		  _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, False)

		  step_time += (time.time() - start_time)
		  loss += step_loss / FLAGS.steps_per_checkpoint
		  current_step += 1
		  #sys.Pause()

		  if current_step % FLAGS.steps_per_checkpoint ==0:
			  # Print statistics for the previous epoch.
			  perplexity = math.exp(loss) if loss < 300 else float('inf')
			  print ("Global step : %d learning rate %.5f step-time %.2f perplexity = %.2f" % (model.global_step.eval(),model.learning_rate.eval(), step_time, perplexity))

			  # Decrease learning rate if no improvement was seen over last 3 times.
			  if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
				  sess.run(model.learning_rate_decay_op)
			  previous_losses.append(loss)

			  #Save checkpoint and zero timer and loss.
			  checkpoint_path = os.path.join(FLAGS.train_dir, "conversation.ckpt")
			  model.saver.save(sess,checkpoint_path, global_step=model.global_step)
			  step_time, loss =0.0, 0.0

			  # Run evals on test set and print their perplexity
			  for bucket_id in xrange(len(_buckets)):
				  if len(test_set[bucket_id]) == 0:
					  print(" eval: empty bucket %d" %(bucket_id))
					  continue
				  encoder_inputs, decoder_inputs, target_weights = model.get_batch(test_set,bucket_id)
				  #print (np.shape(encoder_inputs))
				  _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)
				  eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')

				  print( "test procedure : bucket : %d perplexity : %.2f" % (bucket_id,eval_ppx))
			  sys.stdout.flush()



def decode():
   with tf.Session() as sess:
	   # Create model and load parameters.
	   # second arguments means this model are not Training
	   model = create_model(sess, True)
	   # we decode one sentence at a time
	   model.batch_size = 1

	   # Load vocabularies

	   vocab_path = os.path.join(FLAGS.data_dir,"Word_map.txt")
	   vocab, Q_vocab = data_utils.initialize_vocabulary(vocab_path)


	   while 1:
		   # Get token-ids for the input sentence
		   sys.stdout.write("Input >> ")
		   sys.stdout.flush()
		   sentence = sys.stdin.readline()
		   token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), vocab)
		   #print sentence
		   #print token_ids
		   #print np.shape(token_ids)
		   # Which bucket oes it belong to?
		   bucket_id = min([b for b in xrange(len(_buckets)) if _buckets[b][0] > len(token_ids)])

		   # Get a 1-element batch to feed the sentence to the model.
		   encoder_inputs, decoder_inputs, target_weights = model.get_batch({bucket_id : [(token_ids,[])]},bucket_id)
		   #print np.shape(decoder_inputs)
		   # Get output logits for the sentence.
		   _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)

		   bucket_length = (_buckets[bucket_id])[1]
#		   softmax_output_logits = np.zeros(),dtype=np.float)
		   #outputs = np.zeros(bucket_length,np.int)
		   outputs = []
		   max_outputs = [ int(np.argmax(logit, axis=1)) for logit in output_logits]
		   
		  # for i in range(bucket_length):
		  # 	softmax_output_logits = sess.run(tf.nn.softmax(output_logits[i]))
		  #	cum_sum = np.cumsum(softmax_output_logits)
		  #	random_number_02 = np.random.random_sample()
			#print softmax_output_logits.max()
			#print softmax_output_logits.argmax()
#	  	max_outputs.append(softmax_output_logits.argmax())
		  #	output = min( [j for j in xrange(len(cum_sum)) if cum_sum[j] > random_number_02] )
		  #	outputs.append(output)
		   # This is a greedy decoder - outputs are just argmaxes of output_logits.
#		   outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
		   # If there is an EOS symbol in outputs, cut them at that point.
#
#		   if data_utils.EOS_ID in outputs:
#			   outputs = outputs[:outputs.index(data_utils.EOS_ID)]
		   if data_utils.EOS_ID in max_outputs:
		   	   max_outputs = max_outputs[:max_outputs.index(data_utils.EOS_ID)]
		   #print Q_vocab[outputs[0]]
		   #print (outputs)
#		   print ("sampling output >>")
#		   print (" ".join([tf.compat.as_str(Q_vocab[output]) for output in outputs]))
		
		   #print (max_outputs)
		   print ("output >>")
		   print (" ".join([tf.compat.as_str(Q_vocab[output]) for output in max_outputs]))
		   print("=====================")



def main(argv):
#	print argv
#	if argv == 0:
		train()
#	elif argv == 1:
#	decode()

if __name__ == "__main__":
	tf.app.run()
