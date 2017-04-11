import numpy as np
import tensorflow as tf
import nltk.data
from nltk.stem.snowball import SnowballStemmer
from nltk import word_tokenize
import data_utils
import random, sys

class BaseParseModel(object):
    def __init__(self, config, train_data):
        '''
        Builds the computation graph for the parsing model used
        '''
        self.source_vocab_size = config.source_vocab_size
        self.target_vocab_size = config.target_vocab_size
        self.encoder_size = config.source_max_len
        self.decoder_size = config.target_max_len
        self.batch_size = config.batch_size
        self.train_data = train_data
        self.learning_rate = tf.Variable(float(config.learning_rate), trainable=False)
        self.is_test = tf.placeholder(tf.bool)
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * config.learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)
        self.keep_prob = 1 - config.dropout_rate
        self.keep_prob_input= tf.placeholder(tf.float32) #For dropout control
        self.batch_ind = 0
        self.words_to_id = config.words_to_id
        self.logic_to_id = config.logic_to_id
        self.id_to_words = config.id_to_words
        self.id_to_logic = config.id_to_logic
        self.complete_epoch = False

        # Create LSTM cell
        print("\tCreating Cell")
        single_cell = tf.nn.rnn_cell.LSTMCell(config.layer_size, initializer=tf.random_uniform_initializer(minval=-1*config.initialize_width,maxval=config.initialize_width), state_is_tuple=False)
        single_cell = tf.nn.rnn_cell.DropoutWrapper(single_cell, output_keep_prob=self.keep_prob_input)
        cell = single_cell
        if config.num_layers > 1:
            cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * config.num_layers)
        self.cell = cell
        # Feeds for inputs
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []
        print("\tCreating input feeds")
        for i in xrange(self.encoder_size):
            self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                      name="encoder{0}".format(i)))
        for i in xrange(self.decoder_size):
            self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                      name="decoder{0}".format(i)))
            self.target_weights.append(tf.placeholder(tf.float32, shape=[None],
                                                      name="weight{0}".format(i)))

        #TODO: our loss is different (negative log likelihood rather than negative cross entropy)
        # might want to change in the future
        targets = [self.decoder_inputs[i + 1]
                   for i in xrange(len(self.decoder_inputs) - 1)]

        self.outputs, self.encoder_initial_state, self.encoder_final_state, decoder_states = embedding_attention_seq2seq_states(
            self.encoder_inputs, self.decoder_inputs, cell, self.source_vocab_size, self.target_vocab_size, 
            config.layer_size, feed_previous=self.is_test)
        self.train_decoder_states = decoder_states[1]
        self.test_decoder_states = decoder_states[0]

        self.loss = tf.nn.seq2seq.sequence_loss(self.outputs[:-1], targets, self.target_weights[:-1])

        params = tf.trainable_variables()
        opt = tf.train.AdagradOptimizer(self.learning_rate) 

        print("\tCreating gradients")
        gradients = tf.gradients(self.loss, params)
        clipped_gradients, self.gradient_norm = tf.clip_by_global_norm(gradients, config.max_gradient)
        self.update = opt.apply_gradients(zip(clipped_gradients, params))
        self.saver = tf.train.Saver(tf.all_variables())

    def get_batch(self, is_test, test_data=None):
        """ Gets batch, formats it in correct way.

        If is_test=True, data is assumed to be a 1D list of (input,output) pairs.
           All of this data is added to the final batch.
        Otherwise, the data is sampled from the specified bucket
        """

        if is_test:
            batch_size = len(test_data)
            batch = test_data
            if test_data == None:
                raise Exception("In test mode, data must be provided!")
        else:
            batch_size = self.batch_size
            batch = []
            for _ in xrange(batch_size):
                if self.batch_ind == len(self.train_data):
                    self.batch_ind = 0
                    random.shuffle(self.train_data)
                    batch_size = len(batch)
                    self.complete_epoch = True
                    break
                batch.append(self.train_data[self.batch_ind])
                self.batch_ind += 1
            if self.batch_ind == len(self.train_data):
                self.batch_ind = 0
                random.shuffle(self.train_data)
                self.complete_epoch = True
        num_sentences = max(map(len, batch))
        encoder_inputs, decoder_inputs = [], []
        for i in xrange(num_sentences):
            encoder_inputs.append([])
            decoder_inputs.append([])

        # pad entries if needed, reverse encoder inputs
        for entry in batch:
            for sent_ind in xrange(num_sentences):
                if sent_ind < len(entry):
                    encoder_input = entry[sent_ind][0]
                    decoder_input = entry[sent_ind][1]
                    # Encoder inputs are padded and then reversed
                    encoder_pad = [data_utils.PAD_ID] * (self.encoder_size - len(encoder_input))
                    encoder_inputs[sent_ind].append(list(reversed(encoder_input + encoder_pad)))

                    decoder_pad = [data_utils.PAD_ID] * (self.decoder_size - len(decoder_input))
                    decoder_inputs[sent_ind].append(decoder_input + decoder_pad)
                else:
                    # No sentence here. PAD EVERYTHING
                    encoder_inputs[sent_ind].append([data_utils.PAD_ID] * self.encoder_size)
                    decoder_inputs[sent_ind].append([data_utils.PAD_ID] * self.decoder_size)

        # Now we create batch-major vectors from the data selected above
        final_encoder_inputs, final_decoder_inputs, final_weights = [], [], []
        for sent_ind in xrange(num_sentences):
            batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []
            final_encoder_inputs.append(batch_encoder_inputs)
            final_decoder_inputs.append(batch_decoder_inputs)
            final_weights.append(batch_weights)

            # Batch encoder inputs are just re-indexed encoder_inputs.
            for length_idx in xrange(self.encoder_size):
                batch_encoder_inputs.append(
                    np.array([encoder_inputs[sent_ind][batch_idx][length_idx]
                            for batch_idx in xrange(batch_size)], dtype=np.int32))

            # Batch decoder inputs are re-indexed decoder inputs. We also create weights!
            for length_idx in xrange(self.decoder_size):
                batch_decoder_inputs.append(
                    np.array([decoder_inputs[sent_ind][batch_idx][length_idx]
                              for batch_idx in xrange(batch_size)], dtype=np.int32))

                # Create target_weights to be 0 for targets that are padding
                batch_weight = np.ones(batch_size, dtype=np.float32)
                for batch_idx in xrange(batch_size):
                    # We set weight to 0 if the corresponding target is a PAD symbol.
                    # The corresponding target is decoder_input shifted by 1 forward.
                    if length_idx < self.decoder_size - 1:
                        target = decoder_inputs[sent_ind][batch_idx][length_idx+1]
                    if length_idx == self.decoder_size - 1 or target == data_utils.PAD_ID:
                        batch_weight[batch_idx] = 0.0
                batch_weights.append(batch_weight)
        return batch, final_encoder_inputs, final_decoder_inputs, final_weights

    def step(self, session, is_test, test_data):
        raise NotImplementedError

    def logits2sentences(self, output_logits):
        total_outputs = []
        for sent_ind in xrange(len(output_logits)):
            temp_outputs = [[int(np.argmax(logit)) for logit in output_logit] for output_logit in output_logits[sent_ind]]
            #Reshape outputs
            outputs = np.array(temp_outputs).T.tolist()
      
            for i in xrange(len(outputs)):
                if outputs[i][0] == data_utils.PAD_ID:
                    outputs[i] = None
                elif self.logic_to_id[data_utils.EOS] in outputs[i]:
                    outputs[i] = outputs[i][:outputs[i].index(self.logic_to_id[data_utils.EOS])]
            total_outputs.append(outputs)
        total_outputs =  zip(*total_outputs)
        for entry_ind in xrange(len(total_outputs)):
            total_outputs[entry_ind] = list(total_outputs[entry_ind])
            for sent_ind in xrange(len(total_outputs[entry_ind])):
                total_outputs[entry_ind][sent_ind] = map(lambda x: self.id_to_logic[x], total_outputs[entry_ind][sent_ind])
        return total_outputs
        
    def parse(self, session, text):
        sent_splitter = nltk.data.load('tokenizers/punkt/english.pickle')
        stemmer = SnowballStemmer("english")
        sentences = sent_splitter.tokenize(text)
        bad_list = []
        for sent_ind in xrange(len(sentences)):
            sentences[sent_ind] = word_tokenize(sentences[sent_ind])
            sentences[sent_ind].insert(0, '<s>')
            sentences[sent_ind][-1] = '</s>'
            for word_ind in xrange(len(sentences[sent_ind])):
                stemmed = stemmer.stem(sentences[sent_ind][word_ind])
                if stemmed not in self.words_to_id:
                    bad_list.append(sentences[sent_ind][word_ind])
                else:
                    sentences[sent_ind][word_ind] = self.words_to_id[stemmed]
        if len(bad_list) > 0:
            return False, bad_list
        _, _, logits = self.step(session, True, [zip(sentences, [[self.logic_to_id["<s>"]]]*len(sentences))])
        return True, self.logits2sentences(logits)


'''
This parser model doesn't do anything special - every sentence is
handled completely separately
'''
class ParseModel(BaseParseModel):

    def step(self, session, is_test, test_data=None):
        """
        Run a step of the model feeding the given inputs
        """
        # First, get data in correct format
        batch, encoder_inputs, decoder_inputs, target_weights = self.get_batch(is_test, test_data)

        # Check if the sizes match
        if len(encoder_inputs[0]) != self.encoder_size:
            raise ValueError("Encoder length must be equal to the one in bucket,"
                             " %d != %d." % (len(encoder_inputs[0]), self.encoder_size))
        if len(decoder_inputs[0]) != self.decoder_size:
            raise ValueError("Decoder length must be equal to the one in bucket,"
                             " %d != %d." % (len(decoder_inputs[0]), self.decoder_size))
        if len(target_weights[0]) != self.decoder_size:
            raise ValueError("Weights length must be equal to the one in bucket,"
                             " %d != %d." % (len(target_weights[0]), self.decoder_size))
        outputs = []
        for sent_ind in xrange(len(encoder_inputs)):
            input_feed = {}
            for l in xrange(self.encoder_size):
                input_feed[self.encoder_inputs[l].name] = encoder_inputs[sent_ind][l]
            for l in xrange(self.decoder_size):
                input_feed[self.decoder_inputs[l].name] = decoder_inputs[sent_ind][l]
                input_feed[self.target_weights[l].name] = target_weights[sent_ind][l]

            input_feed[self.is_test] = is_test
            input_feed[self.encoder_initial_state.name] = tf.zeros([len(encoder_inputs[sent_ind][0]), self.cell.state_size]).eval()

            if is_test:
                output_feed = [self.loss]   #Loss for this batch
                for l in xrange(self.decoder_size):  # Output logits
                    output_feed.append(self.outputs[l])
                input_feed[self.keep_prob_input] = 1.0

            else: 
                output_feed = [self.update,  #Update Op that does RMSProp
                               self.gradient_norm,   # Gradient norm
                               self.loss]   #Loss for this batch
                input_feed[self.keep_prob_input] = self.keep_prob
            outputs.append(session.run(output_feed, input_feed))
        if is_test:
            total_loss = sum(map(lambda x: x[0], outputs))
            logits = map(lambda x: x[1:], outputs)
            return batch, total_loss, logits #No gradient norm, loss, outputs
        else:
            total_loss = sum(map(lambda x: x[2], outputs))
            return batch, total_loss, None  # Gradient norm, loss, no outputs

'''
This parser model passes the final hidden states of previous sentences
into the model for the next sentence.
'''
class MultiSentParseModel(BaseParseModel):

    def step(self, session, is_test, test_data=None):
        """
        Run a step of the model feeding the given inputs
        """

        # First, get data in correct format
        batch, encoder_inputs, decoder_inputs, target_weights = self.get_batch(is_test, test_data)

        # Check if the sizes match
        if len(encoder_inputs[0]) != self.encoder_size:
            raise ValueError("Encoder length must be equal to the one in bucket,"
                             " %d != %d." % (len(encoder_inputs[0]), self.encoder_size))
        if len(decoder_inputs[0]) != self.decoder_size:
            raise ValueError("Decoder length must be equal to the one in bucket,"
                             " %d != %d." % (len(decoder_inputs[0]), self.decoder_size))
        if len(target_weights[0]) != self.decoder_size:
            raise ValueError("Weights length must be equal to the one in bucket,"
                             " %d != %d." % (len(target_weights[0]), self.decoder_size))
        outputs = []
        prev_encoder_state = None
        for sent_ind in xrange(len(encoder_inputs)):
            input_feed = {}
            for l in xrange(self.encoder_size):
                input_feed[self.encoder_inputs[l].name] = encoder_inputs[sent_ind][l]
            for l in xrange(self.decoder_size):
                input_feed[self.decoder_inputs[l].name] = decoder_inputs[sent_ind][l]
                input_feed[self.target_weights[l].name] = target_weights[sent_ind][l]

            input_feed[self.is_test] = is_test
            if prev_encoder_state != None:
                input_feed[self.encoder_initial_state] = prev_encoder_state
            else:
                input_feed[self.encoder_initial_state] = tf.zeros([len(encoder_inputs[sent_ind][0]), self.cell.state_size]).eval()
            if is_test:
                output_feed = [self.encoder_final_state, 
                               self.loss]   #Loss for this batch
                for l in xrange(self.decoder_size):  # Output logits
                    output_feed.append(self.outputs[l])
                input_feed[self.keep_prob_input] = 1.0

            else: 
                output_feed = [self.encoder_final_state,
                               self.update,  #Update Op that does RMSProp
                               self.gradient_norm,   # Gradient norm
                               self.loss]   #Loss for this batch
                input_feed[self.keep_prob_input] = self.keep_prob
            outputs.append(session.run(output_feed, input_feed))
            prev_encoder_state = outputs[-1][0]
        if is_test:
            total_loss = sum(map(lambda x: x[1], outputs))
            logits = map(lambda x: x[2:], outputs)
            return batch, total_loss, logits #No gradient norm, loss, outputs
        else:
            total_loss = sum(map(lambda x: x[3], outputs))
            return batch, total_loss, None  # Gradient norm, loss, no outputs

"""
The following is adapted from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/seq2seq.py

"""
from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import rnn
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import seq2seq


def embedding_attention_seq2seq_states(encoder_inputs,
                                       decoder_inputs,
                                       cell,
                                       num_encoder_symbols,
                                       num_decoder_symbols,
                                       embedding_size,
                                       num_heads = 1,
                                       output_projection=None,
                                       feed_previous=False,
                                       dtype=tf.float32,
                                       scope=None,
                                       initial_state_attention=False):
    with variable_scope.variable_scope(
            scope or "embedding_attention_seq2seq_states") as scope:
        # Encoder.
        encoder_initial_state = tf.placeholder(dtype, [None, cell.state_size], "encoder_initial_state")
        encoder_cell = rnn_cell.EmbeddingWrapper(
            cell, embedding_classes=num_encoder_symbols,
            embedding_size=embedding_size)
        encoder_outputs, encoder_state = rnn.rnn(
            encoder_cell, encoder_inputs, initial_state = encoder_initial_state, dtype=dtype)

        # First calculate a concatenation of encoder outputs to put attention on.
        top_states = [array_ops.reshape(e, [-1, 1, cell.output_size])
                      for e in encoder_outputs]
        attention_states = array_ops.concat(1, top_states)

        # Decoder.
        output_size = None
        if output_projection is None:
            cell = rnn_cell.OutputProjectionWrapper(cell, num_decoder_symbols)
            output_size = num_decoder_symbols

        if isinstance(feed_previous, bool):
            raise Exception("feed_previous must be a tensor!")
        # If feed_previous is a Tensor, we construct 2 graphs and use cond.
        def decoder(feed_previous_bool):
            reuse = None if feed_previous_bool else True
            with variable_scope.variable_scope(
                    variable_scope.get_variable_scope(), reuse=reuse) as scope:
                outputs, decoder_state = seq2seq.embedding_attention_decoder(
                        decoder_inputs,
                        encoder_state,
                        attention_states,
                        cell,
                        num_decoder_symbols,
                        embedding_size,
                        num_heads=num_heads,
                        output_size=output_size,
                        output_projection=output_projection,
                        feed_previous=feed_previous_bool,
                        update_embedding_for_previous=False,
                        initial_state_attention=initial_state_attention)
                return outputs, decoder_state
        true_outputs, true_decoder_state = decoder(True)
        false_outputs, false_decoder_state = decoder(False)
        outputs = tf.cond(feed_previous,
                                        lambda: true_outputs,
                                        lambda: false_outputs)
        return outputs, encoder_initial_state, encoder_state, (true_decoder_state, false_decoder_state)

