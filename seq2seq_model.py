# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Sequence-to-sequence model with an attention mechanism."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.models.rnn.translate import data_utils


class Seq2SeqModel(object):
  """Sequence-to-sequence model with attention and for multiple buckets.

  This class implements a multi-layer recurrent neural network as encoder,
  and an attention-based decoder. This is the same as the model described in
  this paper: http://arxiv.org/abs/1412.7449 - please look there for details,
  or into the seq2seq library for complete model implementation.
  This class also allows to use GRU cells in addition to LSTM cells, and
  sampled softmax to handle large output vocabulary size. A single-layer
  version of this model, but with bi-directional encoder, was presented in
    http://arxiv.org/abs/1409.0473
  and sampled softmax is described in Section 3 of the following paper.
    http://arxiv.org/abs/1412.2007
  """

  def __init__(self, source_target_vocab_size, buckets, size,
               num_layers, max_gradient_norm, batch_size, learning_rate,
               learning_rate_decay_factor, num_samples = 512, forward_only=False):
    """Create the model.

    Args:
      source_target_vocab_size: size of the source/target vocabulary.
      buckets: a list of pairs (I, O), where I specifies maximum input length
        that will be processed in that bucket, and O specifies maximum output
        length. Training instances that have inputs longer than I or outputs
        longer than O will be pushed to the next bucket and padded accordingly.
        We assume that the list is sorted, e.g., [(2, 4), (8, 16)].
      size: number of units in each layer of the model.
      num_layers: number of layers in the model.
      max_gradient_norm: gradients will be clipped to maximally this norm.
      batch_size: the size of the batches used during training;
        the model construction is independent of batch_size, so it can be
        changed after initialization if this is convenient, e.g., for decoding.
      learning_rate: learning rate to start with.
      learning_rate_decay_factor: decay learning rate by this much when needed.
      forward_only: if set, we do not construct the backward pass in the model.
    """
    self.source_target_vocab_size = source_target_vocab_size
    self.buckets = buckets
    self.batch_size = batch_size
    self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
    self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
    self.global_step = tf.Variable(0, trainable=False)

    # If we use sampled softmax, we need an output projection.
    output_projection = None
    softmax_loss_function = None
    # Sampled softmax only makes sense if we sample less than vocabulary size.
    if num_samples > 0 and num_samples < self.source_target_vocab_size:
      with tf.device("/cpu:0"):
        w = tf.get_variable("proj_w", [size, self.source_target_vocab_size])
        w_t = tf.transpose(w)
        b = tf.get_variable("proj_b", [self.source_target_vocab_size])
      output_projection = (w, b)

      def sampled_loss(inputs, labels):
        with tf.device("/cpu:0"):
          labels = tf.reshape(labels, [-1, 1])
          return tf.nn.sampled_softmax_loss(w_t, b, inputs, labels, num_samples,
                                            self.target_vocab_size)
      softmax_loss_function = sampled_loss

    # Create the internal multi-layer cell for our RNN.
    single_cell = tf.nn.rnn_cell.BasicLSTMCell(size)
    cell = single_cell
    if num_layers > 1:
      cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers)


  def get_batch(self, data, bucket_id):
    """
       Get a random batch of data from the specified bucket, prepare for step.

       To feed data in step(..) it must be a list of batch-major vectors, while
       data here contains single length-major cases. 
       So the main logic of this function is to re-index data cases to be in the
       proper format for feeding

       Args:
            * data : a tuple of size len(self.buckets) in which each element contains
                     lists of pairs of input and output data that we use to create a batch.
            * bucket_id : integer, which bucket to get the batch for.

       Returns:
            The triple ( encoder_inputs, decoder_inputs, target_weights ) for
            the constructed batch that has proper formant to call step(...) later.
    """
    encoder_size, decoder_size = self.buckets[bucket_id]
    encoder_inputs, decoder_inputs = [], []

    # Get a random batch of encoder and decoder inputs from data,
    # pad them if need, reverse encoder inputs and add GO to decoder.

    for _ in xrange(self.batch_size):
      encoder_input, decoder_input = random.choice(data[bucket_id])

      # Encoder inputs are padded and then reversed.
      # Conversation model ... need to reverse ???
      # --> reverse operation do not performed
      # i.e., 214 532 33323 55 PAD_ID
      encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
      encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))
      encoder_inputs.append(list(encoder_input + encoder_pad))

      # Decoder inputs get an extra "GO" symbol, and are padded then.
      # i.e., GO_ID 323 242 525 123 EOS_ID PAD_ID PAD_ID PAD_ID PAD_ID
      decoder_pad_size = decoder_size - len(decoder_input) - 1
      decoder_inputs.append([data_utils.GO_ID] + decoder_input + [data_utils.PAD_ID]*decoder_pad_size)


    # Now we create batch-major vectors from the data selected above.
    batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

    # Batch encoder inputs are just re-indexed encoder_inputs
    for length_idx in xrange(encoder_size):
      batch_encoder_inputs.append( np.array( [encoder_inputs[batch_idx][length_idx] for batch_idx in xrange(self.batchsize)], dtype=np.int32) )
     

    # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
    # What is the weights?

    for length_idx in xrange(decoder_size):
      batch_decoder_inputs.append( np.array([decoder_intpus[batch_idx][length_idx] for batch_idx in xrange(self.batch_size)], dtype=np.int32) )

      # Create target_weights to be 0 for targets that are padding.
      # weights of padded zero target are 0 
      batch_weight = np.ones(self.batch_size, dtype=np.float32)
      for batch_idx in xrange(self.batch_size):
        if length_idx < decoder_size -1 :
          target = decoder_inputs[batch_idx][length_idx + 1]
        if length_idx == decoder_size -1 or target == data_utils.PAD_ID:
          batch_weight[batch_idx] = 0.0

      batch_weights.append(batch_weight)
    return batch_encoder_inputs, batch_decoder_inputs, batch_weights

















         
       





































