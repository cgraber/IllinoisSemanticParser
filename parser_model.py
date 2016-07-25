import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops, nn_ops, array_ops
import data_utils
import random

class ParseModel(object):


    def __init__(self, config):
        '''
        Builds the computation graph for the parsing model used
        '''

        self.source_vocab_size = config.source_vocab_size
        self.target_vocab_size = config.target_vocab_size
        self.batch_size = config.batch_size
        self.learning_rate = tf.Variable(float(config.learning_rate), trainable=False)
        self.is_test = tf.placeholder(tf.bool)
        #self.learning_rate_decay_op = self.learning_rate.assign(
        #    self.learning_rate * config.learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)
        self.keep_prob = 1 - config.dropout_rate
        self.keep_prob_input= tf.placeholder(tf.float32) #For dropout control
        self.encoder_size = config.input_max_length
        self.decoder_size = config.output_max_length

        # Create LSTM cell
        print("\tCreating Cell")
        single_cell = tf.nn.rnn_cell.LSTMCell(config.layer_size, initializer=tf.random_uniform_initializer(minval=-1*config.initialize_width,maxval=config.initialize_width))
        single_cell = tf.nn.rnn_cell.DropoutWrapper(single_cell, output_keep_prob=self.keep_prob_input)
        cell = single_cell
        if config.num_layers > 1:
            cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * config.num_layers)

        # Feeds for inputs
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []
        print("\tCreating input feeds")
        for i in xrange(self.encoder_size):
            self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                      name="encoder{0}".format(i)))
        for i in xrange(self.decoder_size):
            self.decoder_inputs.append(tf.placeholder(tf.int64, shape=[None],
                                                      name="decoder{0}".format(i)))
            if i < self.decoder_size - 1:
                self.target_weights.append(tf.placeholder(tf.float32, shape=[None],
                                                          name="weight{0}".format(i)))

        targets = [self.decoder_inputs[i + 1]
                   for i in xrange(len(self.decoder_inputs) - 1)]

        #Unused arg is state of decoders at final time step
        self.outputs, _  = tf.nn.seq2seq.embedding_attention_seq2seq(
                self.encoder_inputs, self.decoder_inputs[:-1], cell,
                num_encoder_symbols = self.source_vocab_size,
                num_decoder_symbols = self.target_vocab_size,
                embedding_size = config.layer_size,
                feed_previous=self.is_test)

        self.loss = binary_sequence_loss(
                self.outputs, targets, self.target_weights,
            self.batch_size, self.target_vocab_size)

        params = tf.trainable_variables()
        opt = tf.train.AdagradOptimizer(self.learning_rate) #TODO: Other options?
        print("\tCreating gradients")
        gradients = tf.gradients(self.loss, params)
        clipped_gradients, self.gradient_norm = tf.clip_by_global_norm(gradients,
                                                         config.max_gradient)
        self.update = opt.apply_gradients(
            zip(clipped_gradients, params), global_step=self.global_step)
        self.saver = tf.train.Saver(tf.all_variables()) #TODO: need to figure out how this works w/hyperparameter tuning

    def step(self, session, encoder_inputs, decoder_inputs, target_weights, is_test):
        """
        Run a step of the model feeding the given inputs
        """
        input_feed = {}
        for l in xrange(self.encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        for l in xrange(self.decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            if l < self.decoder_size - 1:
                input_feed[self.target_weights[l].name] = target_weights[l]
        input_feed[self.is_test] = is_test
        if is_test:
            output_feed = [self.loss]   #Loss for this batch
            for l in xrange(self.decoder_size - 1):  # Output logits
                output_feed.append(self.outputs[l])
            input_feed[self.keep_prob_input] = 1.0

        else: 
            output_feed = [self.update,  #Update Op that does RMSProp
                           self.gradient_norm,   # Gradient norm
                           self.loss]   #Loss for this batch
            input_feed[self.keep_prob_input] = self.keep_prob
        outputs = session.run(output_feed, input_feed)
        if is_test:
            return None, outputs[0], outputs[1:] #No gradient norm, loss, outputs
        else:
            return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs

    def get_batch(self, data, is_test):
        """ Gets batch, formats it in correct way.

        If is_test=True, data is assumed to be a 1D list of (input,output) pairs.
           All of this data is added to the final batch.
        Otherwise, the data is sampled from the specified bucket
        """
        encoder_inputs, decoder_inputs = [], []

        if is_test:
            batch_size = len(data)
            batch = data
        else:
            batch_size = self.batch_size
            batch = []
            for _ in xrange(batch_size):
                batch.append(random.choice(data))

        # pad entries if needed, reverse encoder inputs
        for encoder_input, decoder_input in batch:

            # Encoder inputs are padded and then reversed
            encoder_pad = [data_utils.PAD_ID] * (self.encoder_size - len(encoder_input))
            encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

            decoder_pad = [data_utils.PAD_ID] * (self.decoder_size - len(decoder_input))
            decoder_inputs.append(decoder_input + decoder_pad)

        # Now we create batch-major vectors from the data selected above
        batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

        # Batch encoder inputs are just re-indexed encoder_inputs.
        for length_idx in xrange(self.encoder_size):
            batch_encoder_inputs.append(
                np.array([encoder_inputs[batch_idx][length_idx]
                        for batch_idx in xrange(batch_size)], dtype=np.int32))

        # Batch decoder inputs are re-indexed decoder inputs. We also create weights!
        for length_idx in xrange(self.decoder_size):
            batch_decoder_inputs.append(
                np.array([decoder_inputs[batch_idx][length_idx]
                          for batch_idx in xrange(batch_size)], dtype=np.int32))

            # Create target_weights to be 0 for targets that are padding
            batch_weight = np.ones(batch_size, dtype=np.float32)
            for batch_idx in xrange(batch_size):
                # We set weight to 0 if the corresponding target is a PAD symbol.
                # The corresponding target is decoder_input shifted by 1 forward.
                if length_idx < self.decoder_size - 1:
                    target = decoder_inputs[batch_idx][length_idx+1]
                if length_idx == self.decoder_size - 1 or target == data_utils.PAD_ID:
                    batch_weight[batch_idx] = 0.0
            batch_weights.append(batch_weight)
        return batch, batch_encoder_inputs, batch_decoder_inputs, batch_weights

def binary_sequence_loss_by_example(logits, targets, weights,
                         batch_size, num_target_labels,
                         average_across_timesteps=True,
                         softmax_loss_function=None, name=None):
    """Binary loss (full explanation tbd)
    
    Args:
      logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
      targets: List of 1D batch-sized int32 Tensors of the same length as logits.
      weights: List of 1D batch-sized float-Tensors of the same length as logits.
      average_across_timesteps: If set, divide the returned cost by the total
        label weight.
      softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
        to be used instead of the standard softmax (the default if this is None).
      name: Optional name for this operation, default: "binary_sequence_loss".

    Returns:
      1D batch-sized float Tensor: the loss for each sequence

    Raises:
      ValueError: If len(logits) is different from len(targets) or len(weights).
    """
    if len(targets) != len(logits) or len(weights) != len(logits):
        raise ValueError("Lengths of logits, weights, and targets must be the same "
                         "%d, %d, %d." % (len(logits), len(weights), len(targets)))
    with ops.op_scope(logits + targets + weights, name,
                      "binary_sequence_loss"):
        loss_list = []
        for logit, target, weight in zip(logits, targets, weights):
            #Find best guess
            max_ind = tf.argmax(logit, 1)
            #Figure out if correct
            is_correct = tf.equal(max_ind, target)
            is_correct = tf.to_float(is_correct)
            #Depending on whether or not we found the right answer, the loss is different.
            #If correct, use the normal loss
            is_correct_loss = nn_ops.sparse_softmax_cross_entropy_with_logits(logit, target)
            is_correct_loss = tf.mul(is_correct, is_correct_loss)
            #Else, construct new labels where probability distributed evenly among all other classes
            new_target = tf.one_hot(max_ind, num_target_labels, on_value=0.0, off_value = 1.0/(num_target_labels - 1), axis=-1, dtype=tf.float32)
            is_incorrect_loss = nn_ops.softmax_cross_entropy_with_logits(logit, new_target)
            is_incorrect_loss = tf.mul(tf.sub(tf.constant(1.0), is_correct), is_incorrect_loss)

            binary_loss = tf.add(is_correct_loss, is_incorrect_loss)
            loss_list.append(binary_loss * weight)
        losses = math_ops.add_n(loss_list)
        if average_across_timesteps:
            total_size = math_ops.add_n(weights)
            total_size += 1e-12
            losses /= total_size
    return losses

def binary_sequence_loss(logits, targets, weights,
                         batch_size, num_target_labels,
                         average_across_timesteps=True, average_across_batch=True,
                         softmax_loss_function=None, name=None):
    with ops.op_scope(logits + targets + weights, name, "sequence_loss"):
        cost = math_ops.reduce_sum(binary_sequence_loss_by_example(
            logits, targets, weights, batch_size, num_target_labels,
            average_across_timesteps=average_across_timesteps,
            softmax_loss_function=softmax_loss_function))
        if average_across_batch:
            return cost / math_ops.cast(batch_size, tf.float32)
        else:
            return cost
