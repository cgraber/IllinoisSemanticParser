import numpy as np
import tensorflow as tf


class ParseModel(object):


    def __init__(self, config):
        '''
        Builds the computation graph for the parsing model used
        '''

        self.source_vocab_size = config.source_vocab_size
        self.target_vocab_size = config.target_vocab_size
        self.buckets = buckets
        self.batch_size = config.batch_size
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.is_test = tf.placeholder(tf.bool)
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)
        self.keep_prob = 1 - config.dropout_rate
        self.keep_prob_input= tf.placeholder(tf.float32) #For dropout control

        # Create LSTM cell
        single_cell = tf.nn.rnn_cell.LSTMCell(config.layer_size, initializer=tf.random_uniform_initializer(minval=-1*config.initialize_width,maxval=config.initialize_width)
        single_cell = tf.nn.rnn_cell.DropoutWrapper(config.single_cell, output_keep_prob=self.keep_prob_input)
        cell = single_cell
        if config.num_layers > 1:
            cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * config.num_layers)

        # Function to create sequence model
        def seq2seq_f(encoder_inputs, decoder_inputs):
            return tf.nn.seq2seq.embedding_attention_seq2seq(
                encoder_inputs, decoder_inputs, cell,
                num_encoder_symbols = self.source_vocab_size,
                num_decoder_symbols = self.target_vocab_size,
                embedding_size = config.param_size,
                feed_previous=self.is_test)

        # Feeds for inputs
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []
        for i in xrange(buckets[-1][0]):
            self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                      name="encoder{0}".format(i)))
        for i in xrange(buckets[-1][1] + 1):
            self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                      name="decoder{0}".format(i)))
            self.target_weights.append(tf.placeholder(tf.float32, shape=[None],
                                                      name="weight{0}".format(i)))

        #TODO: our loss is different (negative log likelihood rather than negative cross entropy)
        # might want to change in the future
        targets = [self.decoder_inputs[i + 1]
                   for i in xrange(len(self.decoder_inputs) - 1)]

        self.outputs, self.losses = tf.nn.seq2seq.model_with_buckets(
            self.encoder_inputs, self.decoder_inputs, targets,
            self.target_weights, buckets, lambda x, y: seq2seq_f(x, y),
            softmax_loss_function=softmax_loss_function)

        params = tf.trainable_variables()
        self.gradient_norms = []
        self.updates = []
        opt = tf.train.RMSPropOptimizer(self.learning_rate) #TODO: params?
        for b in xrange(len(buckets)):
            gradients = tf.gradients(self.losses[b], params)
            clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                             config.max_gradient)
            self.gradient_norms.append(norm)
            self.updates.append(opt.apply_gradients(
                zip(clipped_gradients, params), global_step=self.global_step))
        self.saver = tf.train.Saver(tf.all_variables()) #TODO: need to figure out how this works w/hyperparameter tuning

    def step(self, session, encoder_inputs, decoder_inputs, target_weights,
             bucket_id, is_test):
        """
        Run a step of the model feeding the given inputs
        """

        # Check if the sizes match
        encoder_size, decoder_size = self.buckets[bucket_id]
        if len(encoder_inputs) != encoder_size:
            raise ValueError("Encoder length must be equal to the one in bucket,"
                             " %d != %d." % (len(encoder_inputs), encoder_size))
        if len(decoder_inputs) != decoder_size:
            raise ValueError("Decoder length must be equal to the one in bucket,"
                             " %d != %d." % (len(decoder_inputs), decoder_size))
        if len(target_weights) != decoder_size:
            raise ValueError("Weights length must be equal to the one in bucket,"
                             " %d != %d." % (len(target_weights), decoder_size))

        input_feed = {}
        for l in xrange(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        for l in xrange(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]

        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)
        input_feed[self.is_test] = is_test

        if is_test:
            output_feed = [self.losses[bucket_id]]   #Loss for this batch
            for l in xrange(decoder_size):  # Output logits
                output_feed.append(self.outputs[bucket_id][l])
            input_feed[self.keep_prob_input] = 1.0
        else: 
            output_feed = [self.updates[bucket_id],  #Update Op that does RMSProp
                           self.gradient_norms[bucket_id],   # Gradient norm
                           self.losses[bucket_id]]   #Loss for this batch
            input_feed[self.keep_prob_input] = self.keep_prob

        outputs = session.run(output_feed, input_feed)
        if is_test:
            return None, outputs[0], outputs[1:] #No gradient norm, loss, outputs
        else:
            return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs

    def get_batch(self, data, bucket_id, is_test, batch_size=self.batch_size):
        """ Gets batch, formats it in correct way.

        If is_test=True, data is assumed to be a 1D list of (input,output) pairs.
           All of this data is added to the final batch.
        Otherwise, the data is sampled from the specified bucket
        """
        encoder_size, decoder_size = self.buckets[bucket_id]
        encoder_inputs, decoder_inputs = [], []

        if is_test:
            batch_size = len(data)
            batch = data
        else:
            batch = []
            for _ in xrange(batch_size):
                batch.append(random.choice(data[bucket_id]))

        # pad entries if needed, reverse encoder inputs
        for encoder_input, decoder_input in batch:

            # Encoder inputs are padded and then reversed
            encoder_pad = [data.PAD_ID] * (encoder_size - len(encoder_input))
            encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

            decoder_pad = [data.PAD_ID] * (decoder_size - len(decoder_input))
            decoder_inputs.append(decoder_input + decoder_pad)

        # Now we create batch-major vectors from the data selected above
        batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

        # Batch encoder inputs are just re-indexed encoder_inputs.
        for length_idx in xrange(encoder_size):
            batch_encoder_inputs.append(
                np.array([encoder_inputs[batch_idx][length_idx]
                        for batch_idx in xrange(batch_size)], dype=np.int32))

        # Batch decoder inputs are re-indexed decoder inputs. We also create weights!
        for length_idx in xrange(decoder_size):
            batch_decoder_inputs.append(
                np.array([decoder_inputs[batch_idx][length_idx]
                          for batch_idx in xrange(batch_size)], dtype=np.int32))

            # Create target_weights to be 0 for targets that are padding
            batch_weight = np.ones(batch_size, dtype=np.float32)
            for batch_idx in xrange(self.batch_size):
                # We set weight to 0 if the corresponding target is a PAD symbol.
                # The corresponding target is decoder_input shifted by 1 forward.
                if length_idx < decoder_size - 1:
                    target = decoder_inputs[batch_idx][length_idx+1]
                if length_idx == decoder_size - 1 or target == data.PAD_ID:
                    batch_weight[batch_idx] = 0.0
            batch_weights.append(batch_weight)
        return batch_encoder_inputs, batch_decoder_inputs, batch_weights
