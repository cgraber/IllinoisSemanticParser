import numpy as np
import tensorflow as tf
import data_utils
import random


class BaseParseModel(object):
    def __init__(self, config):
        self.source_vocab_size = config.source_vocab_size
        self.target_vocab_size = config.target_vocab_size
        self.encoder_size = config.input_max_length
        self.decoder_size = config.output_max_length
        self.batch_size = config.batch_size
        self.learning_rate = tf.Variable(float(config.learning_rate), trainable=False)
        self.is_test = tf.placeholder(tf.bool)
        #self.learning_rate_decay_op = self.learning_rate.assign(
        #    self.learning_rate * config.learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)
        self.keep_prob = 1 - config.dropout_rate
        self.keep_prob_input = tf.placeholder(tf.float32) #For dropout control
        self.num_gpus = config.num_gpus
        self.layer_size = config.layer_size
        self.initialize_width = config.initialize_width
        self.num_layers = config.num_layers

    def build_inference(self):
        # Feeds for inputs
        encoder_inputs = []
        decoder_inputs = []
        target_weights = []
        print("\tCreating input feeds")
        for i in xrange(self.encoder_size):
            encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                      name="encoder{0}".format(i)))
        for i in xrange(self.decoder_size + 1):
            decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                      name="decoder{0}".format(i)))
            target_weights.append(tf.placeholder(tf.float32, shape=[None],
                                                      name="weight{0}".format(i)))
        with tf.device('/cpu:0'):
            print("\tCreating Cell")
            single_cell = tf.nn.rnn_cell.LSTMCell(self.layer_size, initializer=tf.random_uniform_initializer(minval=-1*self.initialize_width,maxval=self.initialize_width))
            single_cell = tf.nn.rnn_cell.DropoutWrapper(single_cell, output_keep_prob=self.keep_prob_input)
            cell = single_cell
            if self.num_layers > 1:
                cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * self.num_layers)

            outputs, _ = tf.nn.seq2seq.embedding_attention_seq2seq(
                encoder_inputs, decoder_inputs, cell,
                num_encoder_symbols = self.source_vocab_size,
                num_decoder_symbols = self.target_vocab_size,
                embedding_size = self.layer_size,
                feed_previous = self.is_test)

        target = [decoder_inputs[i+1]
            for i in xrange(len(decoder_inputs) - 1)]
        return encoder_inputs, decoder_inputs, target, target_weights[:-1], outputs[:-1]



    def build_loss(self, logits, targets, weights):
        losses = tf.nn.seq2seq.sequence_loss(logits, targets, weights)
        return losses
        
    def get_batch(self, data, is_test):
        """ Gets batch, formats it in correct way.

        If is_test=true, all of the data is used. Otherwise, a batch of size
        batch_size is sampled.
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

        # NOw we create batch-major vectors from the data selected above
        batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

        # Batch encoder inputs are just re-indexed encoder_inputs.
        for length_idx in xrange(self.encoder_size):
            batch_encoder_inputs.append(
                np.array([encoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(batch_size)], dtype=np.int32))

        # Batch decoder inputs are just re-indexed decoder inputs. We also create weights!
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


class ParseModel(object):


    def __init__(self, config):
        '''
        Builds the computation graph for the parsing model used
        '''

        self.source_vocab_size = config.source_vocab_size
        self.target_vocab_size = config.target_vocab_size
        self.buckets = config.buckets
        self.batch_size = config.batch_size
        self.learning_rate = tf.Variable(float(config.learning_rate), trainable=False)
        self.is_test = tf.placeholder(tf.bool)
        #self.learning_rate_decay_op = self.learning_rate.assign(
        #    self.learning_rate * config.learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)
        self.keep_prob = 1 - config.dropout_rate
        self.keep_prob_input= tf.placeholder(tf.float32) #For dropout control

        # Create LSTM cell
        print("\tCreating Cell")
        single_cell = tf.nn.rnn_cell.LSTMCell(config.layer_size, initializer=tf.random_uniform_initializer(minval=-1*config.initialize_width,maxval=config.initialize_width))
        single_cell = tf.nn.rnn_cell.DropoutWrapper(single_cell, output_keep_prob=self.keep_prob_input)
        cell = single_cell
        if config.num_layers > 1:
            cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * config.num_layers)

        # Function to create sequence model
        def seq2seq_f(encoder_inputs, decoder_inputs):
            return tf.nn.seq2seq.embedding_attention_seq2seq(
                encoder_inputs, decoder_inputs, cell,
                num_encoder_symbols = self.source_vocab_size,
                num_decoder_symbols = self.target_vocab_size,
                embedding_size = config.layer_size,
                feed_previous=self.is_test)

        # Feeds for inputs
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []
        print("\tCreating input feeds")
        for i in xrange(self.buckets[-1][0]):
            self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                      name="encoder{0}".format(i)))
        for i in xrange(self.buckets[-1][1] + 1):
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
            self.target_weights, self.buckets, lambda x, y: seq2seq_f(x, y))

        params = tf.trainable_variables()
        self.gradient_norms = []
        self.updates = []
        opt = tf.train.AdagradOptimizer(self.learning_rate) #TODO: Other options?
        print("\tCreating gradients")
        for b in xrange(len(self.buckets)):
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
        input_feed[last_target] = np.zeros([len(decoder_inputs[0])], dtype=np.int32)
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

    def get_batch(self, data, bucket_id, is_test):
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
            batch_size = self.batch_size
            batch = []
            for _ in xrange(batch_size):
                batch.append(random.choice(data[bucket_id]))

        # pad entries if needed, reverse encoder inputs
        for encoder_input, decoder_input in batch:

            # Encoder inputs are padded and then reversed
            encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
            encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

            decoder_pad = [data_utils.PAD_ID] * (decoder_size - len(decoder_input))
            decoder_inputs.append(decoder_input + decoder_pad)

        # Now we create batch-major vectors from the data selected above
        batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

        # Batch encoder inputs are just re-indexed encoder_inputs.
        for length_idx in xrange(encoder_size):
            batch_encoder_inputs.append(
                np.array([encoder_inputs[batch_idx][length_idx]
                        for batch_idx in xrange(batch_size)], dtype=np.int32))

        # Batch decoder inputs are re-indexed decoder inputs. We also create weights!
        for length_idx in xrange(decoder_size):
            batch_decoder_inputs.append(
                np.array([decoder_inputs[batch_idx][length_idx]
                          for batch_idx in xrange(batch_size)], dtype=np.int32))

            # Create target_weights to be 0 for targets that are padding
            batch_weight = np.ones(batch_size, dtype=np.float32)
            for batch_idx in xrange(batch_size):
                # We set weight to 0 if the corresponding target is a PAD symbol.
                # The corresponding target is decoder_input shifted by 1 forward.
                if length_idx < decoder_size - 1:
                    target = decoder_inputs[batch_idx][length_idx+1]
                if length_idx == decoder_size - 1 or target == data_utils.PAD_ID:
                    batch_weight[batch_idx] = 0.0
            batch_weights.append(batch_weight)
        return batch, batch_encoder_inputs, batch_decoder_inputs, batch_weights

class MultiParseModel(BaseParseModel):
    def __init__(self, config):
        with tf.device('/cpu:0'):
            super(MultiParseModel,self).__init__(config)

        opt = tf.train.AdagradOptimizer(self.learning_rate) #TODO: Other options?
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []
        self.outputs = []
        tower_grads = []
        self.losses = []
        print("\tConstructing gpu graphs")
        for i in xrange(config.num_gpus):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('tower_%d' % i) as scope:
                    print("\t\tConstructing graph for gpu %d"%i)
                    # Calculate the loss for one tower of this model. This function
                    # constructs the entire model but shares the variables across
                    # all towers.
                    encoder_input, decoder_input, target, target_weight, output = self.build_inference()
                    loss = self.build_loss(output, target, target_weight)
                    self.losses.append(loss)
                    self.encoder_inputs.append(encoder_input)
                    self.decoder_inputs.append(decoder_input)
                    self.target_weights.append(target_weight)
                    self.outputs.append(output)

                    # Reuse variables for the next tower.
                    tf.get_variable_scope().reuse_variables()

                    grads_and_vars = opt.compute_gradients(loss)
                    clipped_grads_and_vars = [(tf.clip_by_global_norm(gv[0],config.max_gradient)[0], gv[1]) 
                            for gv in grads_and_vars]
                    tower_grads.append(clipped_grads_and_vars)

        grads = average_gradients(tower_grads)
        self.update_op = opt.apply_gradients(grads, global_step=global_step)
        self.total_loss = tf.add_n(self.losses)

        saver = tf.train.Saver(tf.all_variables())

    def average_gradients(tower_grads):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #    ((grad0_gpu0, var0_gpu0), ..., (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(0, grads)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared across
            # towers. So .. we will just return the first tower's pointer to
            # the Variable
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

    def step(self, session, data, is_test):
        input_feed = {}
        input_feed[self.is_test] = is_test
        if is_test:
            encoder_inputs, decoder_inputs, target_weights = self.get_batch(data, is_test)
            output_feed = [self.losses[0]]
            for l in xrange(self.encoder_size):
                input_feed[self.encoder_inputs[0][l].name] = encoder_inputs[l]
            for l in xrange(self.decoder_size):
                input_feed[self.decoder_inputs[0][l].name] = decoder_inputs[l]
                if l < self.decoder_size - 1:
                    input_feed[self.target_weights[0][l].name] = target_weights[l]
                    output_feed.append(self.outputs[0][l])
            input_feed[self.keep_prob_input] = 1.0
        else:
            for i in xrange(self.num_gpus):
                encoder_inputs, decoder_inputs, target_weights = self.get_batch(data, is_test)
                for l in xrange(self.encoder_size):
                    input_feed[self.encoder_inputs[i][l].name] = encoder_inputs[l]
                for l in xrange(self.decoder_size):
                    input_feed[self.decoder_inputs[i][l].name] = decoder_inputs[l]
                    input_feed[self.target_weights[i][l].name] = target_weights[l]
            output_feed = [self.update_op,
                           sel.total_loss]
            input_feed[self.keep_prob_input] = self.keep_prob
        outputs = session.run(output_feed, input_feed)
        if is_test:
            return outputs[0], outputs[1:]
        else:
            return outputs[1], None


