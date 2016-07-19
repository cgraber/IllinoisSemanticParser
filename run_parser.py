from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import os, sys, time, random, argparse, pickle
import config, data_utils, parser_model

parser = argparse.ArgumentParser()
parser.add_argument('-lr', '--learning_rate', type=float, default=0.5,
                    help='Learning rate.')
parser.add_argument('-lrdf', '--learning_rate_decay_factor', type=float, default=0.99,
                    help='Learning rate decays by this much.')
parser.add_argument('-mgn', '--max_gradient-norm', type=float, default=5.0,
                    help='Clip gradients to this norm.')
parser.add_argument('-b', '--batch_size', type=int, default=20,
                    help='Batch size to use during training.')
parser.add_argument('-nl', '--num_layers', type=int, default=1,
                    help='Number of layers in the model.')
parser.add_argument('-dd', '--data_dir', default="data/Geo",
                    help='Data directory')
parser.add_argument('-td', '--train_dir', default="./tmp",
                    help='Training directory')
parser.add_argument('-mtds', '--max_train_data_size', type=int, default=0,
                    help='Limit on the size of training data (0: no limit).')
parser.add_argument('-spc', '--steps_per_checkpoint', type=int, default=50,
                    help='How many training steps to do per checkpoint.')
parser.add_argument('-esp', '--early_stopping_patience', type=int, default=500,
                    help='How many rounds to wait until early stopping is enforced.')
parser.add_argument('-nf', '--num_folds', type=int, default=10,
                    help='Number of folds for cross-validation')
parser.add_argument('mode', choices=['train', 'test'],
                    help='Way to run the app')
FLAGS = parser.parse_args()

GEO_BUCKETS = [(10,15), (15,20), (20,25), (40,70)] 
BLOCKS_BUCKETS = [(15,25),(50,100),(100,200)]

_buckets = GEO_BUCKETS

def load_data():
    train_data, test_data, vocab_size = data_utils.load_raw_text(FLAGS.data_dir)
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

def find_bucket(data):
    """Find the smallest bucket holding all of the given data"""
    for i, (source_size, target_size) in enumerate(_buckets):
        works = True
        for entry in data:
            if len(entry[0]) >= source_size and len(entry[1]) >= target_size:
                works = False
                break
        if works:
            return i
    raise Exception("NO BUCKET COULD HOLD ALL OF THE DATA")

def create_model(session, conf):
    """Create model and initialize or load parameters in session."""
    model = parser_model.ParseModel(conf)
    print("Created model with fresh parameters.")
    session.run(tf.initialize_all_variables())
    return model

def train(sess, train_data, validation_data, conf, num_steps = None):
    print("Preparing model...")
    model = create_model(sess, conf)    
    train_buckets = data2buckets(train_data)
    train_bucket_sizes = [len(train_buckets[b]) for b in xrange(len(_buckets))]
    train_total_size = float(sum(train_bucket_sizes))
    checkpoint_dir = os.path.join(FLAGS.train_dir, conf.get_dir())
    checkpoint_path = os.path.join(checkpoint_dir, "parse.ckpt")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size of i-th training bucket, as used later.
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in xrange(len(train_bucket_sizes))]
    print(train_buckets_scale)
    # The training loop.
    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []
    best_validation_loss = float("inf")
    best_validation_acc = 0.0
    best_validation_step = 0
    print("Starting training")
    sys.stdout.flush()
    while not num_steps or current_step < num_steps:
        # Choose a bucket according to data distribution. We pick a random number
        # in [0, 1] and use the corresponding interval in train_buckets_scale.
        random_number_01 = np.random.random_sample()
        bucket_id = min([i for i in xrange(len(train_buckets_scale))
                         if train_buckets_scale[i] > random_number_01])

        # Get a batch and make a step.
        start_time = time.time()
        entries, encoder_inputs, decoder_inputs, target_weights = model.get_batch(
            train_buckets, bucket_id, False)
        _, step_loss, step_outputs = model.step(sess, encoder_inputs, decoder_inputs,
                                     target_weights, bucket_id, False)
        step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
        loss += step_loss / FLAGS.steps_per_checkpoint
        current_step += 1

        # Once in a while, we save checkpoint, print statistics, and run evals
        if current_step % FLAGS.steps_per_checkpoint == 0:

            if not num_steps:
                # Check early stopping condition
                #print("LAST BATCH:")
                temp_loss, temp_acc = test(sess, entries, model)
                #print("VALIDATION:")
                validation_loss, validation_acc = test(sess, validation_data, model)

                print("TEST: PREV LOSS=%.2f, NEW LOSS=%.2f, acc=%.2f"%(step_loss, temp_loss, temp_acc))
                print("global step %s learning rate %.4f step-time %.2f training loss %.2f" %
                    (model.global_step.eval(), model.learning_rate.eval(),
                     step_time, step_loss))
                print("               validation loss %.2f validaiton acc %.2f"%(validation_loss, validation_acc))
                if validation_acc > best_validation_acc or (validation_acc == best_validation_acc and validation_loss < best_validation_loss):
                    best_validation_loss = validation_loss
                    best_validation_step = current_step
                    best_validation_acc = validation_acc
                    model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                if current_step - best_validation_step >= FLAGS.early_stopping_patience:
                    print("Early stopping triggered. Restoring previous model")
                    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
                    model.saver.restore(sess, ckpt.model_checkpoint_path)
                    return model, current_step
            else:
                print("\tIteration %d of %d"%(current_step, num_steps))
            # Decrease learning rate if no improvement was seen over last 3 times.
            #if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
            #    sess.run(model.learning_rate_decay_op)
            previous_losses.append(loss)
            # Save checkpoint and zero timer and loss
            step_time, loss = 0.0, 0.0
            sys.stdout.flush()
    return model, num_steps

def test(sess, test_data, model, dump_results=False):
    test_bucket = find_bucket(test_data)
    _, encoder_inputs, decoder_inputs, target_weights = model.get_batch(
            test_data, test_bucket, True)
    _, loss, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                  target_weights, test_bucket, True)
    return loss, evaluate_logits(output_logits, test_data, dump_results)

def evaluate_logits(output_logits, test_data, dump_results=False):
    temp_outputs = [[int(np.argmax(logit)) for logit in output_logit] for output_logit in output_logits]

    #Reshape outputs
    outputs = np.array(temp_outputs).T.tolist()
  
    for i in xrange(len(outputs)):
        if data_utils.LOGIC_EOS_ID in outputs[i]:
            outputs[i] = outputs[i][:outputs[i].index(data_utils.LOGIC_EOS_ID)]
    
    #print("CORRECT OUTPUTS:")
    #print(data_utils.ids_to_logics(test_data[0][1][1:-1]))
    #print("GIVEN OUTPUTS")
    #print(data_utils.ids_to_logics(outputs[0]))
    if dump_results:
        print("==============TEST FAILURES====================")
    correct = 0.0
    for i in xrange(len(test_data)):
        if test_data[i][1][1:-1] == outputs[i]: #TODO: make sure this is correct
            correct += 1.0
        if dump_results:
            print(' '.join(data_utils.ids_to_words(test_data[i][0])))
            print("\tCorrect: "+''.join(data_utils.ids_to_logics(test_data[i][1][1:-1])))
            print("\tFound:   "+''.join(data_utils.ids_to_logics(outputs[0])))
    return correct/len(test_data)


def cross_validate(splits, conf):
    performance = 0
    for i in xrange(len(splits)):
        print("===================Beginning split %d========================"%i)
        conf.fold = i
        train_data = sum(splits[:i] + splits[i+1:], [])
        validation_data = splits[i]
        with tf.Session() as sess:
            model,_ = train(sess, train_data, validation_data, conf)
            loss, acc = test(sess, validation_data, model)
            performance += loss
        tf.reset_default_graph()
    return performance/len(splits)

def parameter_tuning(folds, source_vocab_size, target_vocab_size):
    best_loss = None
    best_config = None
    for conf in config.config_beam_search(source_vocab_size, target_vocab_size, FLAGS.num_layers, FLAGS.batch_size, _buckets, FLAGS.learning_rate, FLAGS.learning_rate_decay_factor):
        print("+++++++++++++++++++++++Beginning cross-validation with dropout_rate = %0.1f, vector_size=%d++++++++++++++++++"%
               (conf.dropout_rate, conf.layer_size))
        loss = cross_validate(folds, conf)
        if not best_loss or loss < best_loss:
            best_loss = loss
            best_config = conf
    best_config.fold = None
    print("Best config:")
    print("\tdropout: %.1f, param size: %d"%(best_config.dropout_size, best_config.layer_size))
    return best_config

def main_train():
    folds, test_data, (source_vocab_size, target_vocab_size) = load_data()
    train_data = sum(folds[:-1],[])
    validation_data = folds[-1]
    #conf = parameter_tuning(folds, source_vocab_size, target_vocab_size)
    conf = list(config.config_beam_search(source_vocab_size, target_vocab_size, FLAGS.num_layers, FLAGS.batch_size, _buckets, FLAGS.learning_rate, FLAGS.learning_rate_decay_factor))[0]

    #First, train with held-out data to find number of iterations
    with tf.Session() as sess:
        model, num_steps = train(sess, train_data, validation_data, conf)

    #Now train on full data set
    tf.reset_default_graph()
    train_data += validation_data
    with tf.Session() as sess:
        model, _ = train(sess, train_data, None, conf, num_steps)
        model_path = os.path.join(FLAGS.train_dir, 'final_model')
        model.saver.save(sess, model_path)
        conf_out = open(os.path.join(FLAGS.train_dir, 'final_model.conf'), 'w')
        pickle.dump(conf, conf_out)
        conf_out.close()
        loss, acc = test(sess, test_data, model)
        print("FINAL RESULTS:")
        print("  loss = %0.4f"%loss)
        print("  acc  = %0.4f"%acc)

def main_test():
    _, test_data, (source_vocab_size, target_vocab_size) = load_data()
    test_conf_path = os.path.join(FLAGS.train_dir, 'final_model.conf')
    conf_in = open(test_conf_path, 'r')
    conf = pickle.load(conf_in)
    conf_in.close()
    with tf.Session() as sess:
        model = create_model(sess, conf)
        model.saver.restore(sess, os.path.join(FLAGS.train_dir, 'final_model'))
        loss, acc = test(sess, test_data, model, True)
        print("FINAL RESULTS:")
        print("  loss = %0.4f"%loss)
        print("  acc  = %0.4f"%acc)


def main(_):
    if FLAGS.mode == "train":
        main_train()
    else:
        main_test()

if __name__ == "__main__":
    tf.app.run()
