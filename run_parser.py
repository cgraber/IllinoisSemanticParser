from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import config, data, parser_model

tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 20,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")
tf.app.flags.DEFINE_string("data_dir", "data/Geo", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "/tmp", "Training directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 20,
                             "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_integer("early_stopping_patience", 200,
                            "How many rounds to wait until early stopping enforced")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_integer("num_folds", 10,
                            "Number of folds for cross-validation")

FLAGS = tf.app.flags.FLAGS

_buckets = [(10,15), (15,20), (20,25), (40,70)] #TODO: what buckets make sense?

def load_data():
    train_data, test_data, vocab_size = data.load_raw_text(FLAGS.data_dir)
    folds = []
    fold_size = int(len(train_data)/FLAGS.num_folds)
    for i in xrange(FLAGS.num_folds - 1):
        folds.append(train_data[i*fold_size:(i+1)*fold_size])
    folds.append(train_data[(FLAGS.num_folds-1)*fold_size:])
    return folds, test_data, vocab_size

def data2buckets(data):
    bucket_data = [[] for _ in _buckets]
    for entry in data:
        found = False
        for bucket_id, (source_size, target_size) in enumerate(_buckets):
            if len(entry[0]) < source_size and len(entry[1]) < target_size:
                bucket_data[bucket_id].append(entry)
                found = True
                break
        if not found:
            raise Exception("BUCKETS NOT LARGE ENOUGH: (%d, %d)"%(len(entry[0]), len(entry[1])))
    return bucket_data


def create_model(session, conf, is_training):
    """Create model and initialize or load parameters in session."""
    model = parser_model.ParseModel(conf, is_training)
    ckpt = tf.train.get_checkpoint_state(os.path.join(FLAGS.train_dir, conf.get_dir()))
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_apth):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.initialize_all_variables())
    return model

def train(sess, train_data, conf):
    print("Preparing model...")
    model = create_model(sess, conf, True)
    
    train_buckets = read_data(train_data)
    train_bucket_sizes = [len(train_buckets[b]) for b in xrange(len(_buckets))]
    train_total_size = float(sum(train_bucket_sizes))

    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size of i-th training bucket, as used later.
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in xrange(len(train_bucket_sizes))]

    # The training loop.
    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []
    while True:
        # Choose a bucket according to data distribution. We pick a random number
        # in [0, 1] and use the corresponding interval in train_buckets_scale.
        random_number_01 = np.random.random_sample()
        bucket_id = min([i for i in xrange(len(train_buckets_scale))
                         if train_buckets_scale[i] > random_number_01])

        # Get a batch and make a step.
        start_time = time.time()
        encoder_inputs, decoder_inputs, target_weights = model.get_batch(
            train_buckets, bucket_id)
        _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                     target_weights, bucket_id, False)
        step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
        loss += step_loss / FLAGS.steps_per_checkpoint
        current_step += 1

        # Once in a while, we save checkpoint, print statistics, and run evals
        if current_step % FLAGS.steps_per_checkpoint == 0:
            # Print statistics for the previous epoch
            print("global step %s learning rate %.4f step-time %.2f training loss %.2f" %
                (model.global_step.eval(), model.learning_rate.eval(),
                 step_time, loss))

            # Check early stopping condition
            _, loss, _ = model.step(sess, encoder_inputs, decoder_inputs,


            
            # Decrease learning rate if no improvement was seen over last 3 times.
            if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                sess.run(model.learning_rate_decay_op)
            previous_losses.append(loss)
            # Save checkpoint and zero timer and loss
            checkpoint_path = os.path.join(FLAGS.train_dir, conf.get_dir(), "parse.ckpt")
            model.saver.save(sess, checkpoint_path, global_step=model.global_step)
            step_time, loss = 0.0, 0.0
            sys.stdout.flush()

def test(sess, test_data, model):
    for entry in test_data:
        bucket_id = min([b for b in xrange(len(_buckets))
                         if _buckets[b][0] > len(entry[0])])
        # Get a 1-element batch to feed the sentence to the model.
        encoder_inputs, decoder_inputs, target_weights = model.get_batch(
            {bucket_id: [(entry[0], [])]}, bucket_id)
        _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                         target_weights, bucket_id, True)
        outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
        if data_utils.EOS_ID in outputs:
            outputs = outputs[:outputs.index(data.LOGIC_EOS_ID)]
        #TODO: FINISH!

def cross_validate(splits, conf):
    performance = 0
    for i in xrange(len(splits)):
        conf.fold = i
        train_data = sum(splits[:i] + splits[i+1:], [])
        tune_data = splits[i]
        with tf.Session() as sess:
            model = train(sess, train_data, conf)
            performance += test(sess, tune_data, conf, model)
    return performance/len(splits)

def parameter_tuning(splits, source_vocab_size, target_vocab_size):
    best_result = None
    best_config = None
    for conf in config.config_beam_search(source_vocab_size, target_vocab_size):
        result = cross_validate(splits, conf)
        if not best_result or result > best_result: #TODO: Make sure this is correct!
            best_result = result
            best_config = conf
    best_config.fold = None
    return train(sum(splits, []), best_config)

if __name__ == "__main__":
    
    """
    init_op = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init_op)
        
        model = parameter_tuning(splits, vocab_size[0], vocab_size[1])
        result = test(test_data, model)
    """
