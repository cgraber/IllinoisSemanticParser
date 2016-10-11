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
parser.add_argument('domain', choices=['geo', 'blocks'],
                    help='The domain on which to run')
FLAGS = parser.parse_args()

GEO_BUCKETS = [(10,15), (15,20), (20,25), (40,70)] 
#BLOCKS_BUCKETS = [(15,25),(50,100),(100,200)]
BLOCKS_BUCKETS = [(30,75)]

if FLAGS.domain == 'geo':
    _buckets = GEO_BUCKETS
elif FLAGS.domain == 'blocks':
    _buckets = BLOCKS_BUCKETS

def load_data():
    train_data, test_data, vocab_size = data_utils.load_raw_text(FLAGS.data_dir)
    source_max_len, target_max_len = 0,0
    for entry in train_data+test_data:
        for sent in entry:
            source_max_len = max(source_max_len, len(sent[0]))
            target_max_len = max(target_max_len, len(sent[1]))
    folds = []
    fold_size = int(len(train_data)/FLAGS.num_folds)
    for i in xrange(FLAGS.num_folds - 1):
        folds.append(train_data[i*fold_size:(i+1)*fold_size])
    folds.append(train_data[(FLAGS.num_folds-1)*fold_size:])
    return folds, test_data, vocab_size, source_max_len, target_max_len

def create_model(session, conf, train_data):
    """Create model and initialize or load parameters in session."""
    model = parser_model.MultiSentParseModel(conf, train_data)
    print("Created model with fresh parameters.")
    session.run(tf.initialize_all_variables())
    return model

def train(sess, train_data, validation_data, conf, num_steps = None):
    print("Preparing model...")
    model = create_model(sess, conf, train_data)    
    checkpoint_dir = os.path.join(FLAGS.train_dir, conf.get_dir())
    checkpoint_path = os.path.join(checkpoint_dir, "parse.ckpt")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # The training loop.
    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []
    best_validation_loss = float("inf")
    best_validation_acc = 0.0
    best_validation_sentence_acc = 0.0
    best_validation_step = 0
    print("Starting training")
    sys.stdout.flush()
    perfect_count = 0
    while not num_steps or current_step < num_steps:

        # Get a batch and make a step.
        start_time = time.time()
        entries, step_loss, step_outputs = model.step(sess, False)
        step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
        loss += step_loss / FLAGS.steps_per_checkpoint
        current_step += 1

        # Once in a while, we save checkpoint, print statistics, and run evals
        if current_step % FLAGS.steps_per_checkpoint == 0:

            if not num_steps:
                # Check early stopping condition
                #print("LAST BATCH:")
                temp_loss, temp_total_acc, temp_sentence_acc = test(sess, entries, model)
                #print("VALIDATION:")
                validation_loss, validation_total_acc, validation_sentence_acc = test(sess, validation_data, model)
                if validation_sentence_acc == 1.0:
                    perfect_count += 1
                else:
                    perfect_count = 0

                print("global step %s learning rate %.4f step-time %.2f training loss %.4f validation loss %.4f validation total acc %.4f validation sent acc %.4f" %
                    (current_step, model.learning_rate.eval(),
                     step_time, step_loss, validation_loss, validation_total_acc, validation_sentence_acc))
                if validation_sentence_acc > best_validation_sentence_acc or (validation_sentence_acc == best_validation_sentence_acc and validation_loss < best_validation_loss):
                    best_validation_loss = validation_loss
                    best_validation_step = current_step
                    best_validation_acc = validation_total_acc
                    best_validation_sentence_acc = validation_sentence_acc
                    model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                if current_step - best_validation_step >= FLAGS.early_stopping_patience or perfect_count == 5:
                    print("Early stopping triggered. Restoring previous model")
                    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
                    model.saver.restore(sess, ckpt.model_checkpoint_path)
                    return model, best_validation_step
            else:
                print("\tIteration %d of %d"%(current_step, num_steps))
            # Decrease learning rate if no improvement was seen over last 3 times.
            if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                sess.run(model.learning_rate_decay_op)
            previous_losses.append(loss)
            if len(previous_losses) > 3:
                del previous_losses[0]
            # Save checkpoint and zero timer and loss
            step_time, loss = 0.0, 0.0
            sys.stdout.flush()
    return model, num_steps

def test(sess, test_data, model, dump_results=False):
    _, loss, output_logits = model.step(sess, True, test_data)
    total_acc, sentence_acc = evaluate_logits(output_logits, test_data, dump_results)
    return loss, total_acc, sentence_acc

def evaluate_logits(output_logits, test_data, dump_results=False):
    total_outputs = []
    for sent_ind in xrange(len(output_logits)):
        temp_outputs = [[int(np.argmax(logit)) for logit in output_logit] for output_logit in output_logits[sent_ind]]
        #Reshape outputs
        outputs = np.array(temp_outputs).T.tolist()
  
        for i in xrange(len(outputs)):
            if outputs[i][0] == data_utils.PAD_ID:
                outputs[i] = None
            elif data_utils.LOGIC_EOS_ID in outputs[i]:
                outputs[i] = outputs[i][:outputs[i].index(data_utils.LOGIC_EOS_ID)]
        total_outputs.append(outputs)
    total_outputs = zip(*total_outputs)
    
    #print("CORRECT OUTPUTS:")
    #print(data_utils.ids_to_logics(test_data[0][1][1:-1]))
    #print("GIVEN OUTPUTS")
    #print(data_utils.ids_to_logics(outputs[0]))
    if dump_results:
        print("==============TEST FAILURES====================")
    total_correct = 0.0
    sentence_correct = 0.0
    num_sentences = 0.0
    num_entries = 0.0
    for entry_ind in xrange(len(test_data)):
        #print("ENTRY: %d"%entry_ind)
        num_entries += 1.0
        all_correct = True
        for sent_ind in xrange(len(test_data[entry_ind])):
            num_sentences += 1.0
            
            if test_data[entry_ind][sent_ind][1][1:-1] != total_outputs[entry_ind][sent_ind]: #TODO: make sure this is correct
                all_correct = False
            else:
                sentence_correct += 1.0
        if all_correct:
            total_correct += 1.0
        elif dump_results:
            for sent_ind in xrange(len(test_data[entry_ind])):
                print(' '.join(data_utils.ids_to_words(test_data[entry_ind][sent_ind][0])))
                print("\tCorrect: "+''.join(data_utils.ids_to_logics(test_data[entry_ind][sent_ind][1][1:-1])))
                print("\tFound:   "+''.join(data_utils.ids_to_logics(total_outputs[entry_ind][sent_ind])))
    return total_correct/num_entries, sentence_correct/num_sentences


def cross_validate(splits, conf):
    performance = 0
    for i in xrange(len(splits)):
        print("===================Beginning split %d========================"%i)
        conf.fold = i
        train_data = sum(splits[:i] + splits[i+1:], [])
        validation_data = splits[i]
        with tf.Session() as sess:
            model,_ = train(sess, train_data, validation_data, conf)
            loss, total_acc, sentence_acc = test(sess, validation_data, model)
            performance += loss
        tf.reset_default_graph()
    return performance/len(splits)

def parameter_tuning(folds, source_vocab_size, target_vocab_size, source_max_len, target_max_len):
    best_loss = None
    best_config = None
    for conf in config.config_beam_search(source_vocab_size, target_vocab_size, FLAGS.num_layers, FLAGS.batch_size, FLAGS.learning_rate, FLAGS.learning_rate_decay_factor, source_max_len, target_max_len, data_utils.words_to_id, data_utils.logic_to_id, data_utils.id_to_words, data_utils.id_to_logic):
        print("+++++++++++++++++++++++Beginning cross-validation with dropout_rate = %0.1f, vector_size=%d++++++++++++++++++"%
               (conf.dropout_rate, conf.layer_size))
        loss = cross_validate(folds, conf)
        if not best_loss or loss < best_loss:
            best_loss = loss
            best_config = conf
    best_config.fold = None
    print("Best config:")
    print("\tdropout: %.1f, param size: %d"%(best_config.dropout_rate, best_config.layer_size))
    return best_config

def main_train():
    folds, test_data, (source_vocab_size, target_vocab_size), source_max_len, target_max_len = load_data()
    train_data = sum(folds[:-1],[])
    validation_data = folds[-1]
    #conf = parameter_tuning(folds, source_vocab_size, target_vocab_size)
    conf = list(config.config_beam_search(source_vocab_size, target_vocab_size, FLAGS.num_layers, FLAGS.batch_size, FLAGS.learning_rate, FLAGS.learning_rate_decay_factor, source_max_len, target_max_len, data_utils.words_to_id, data_utils.logic_to_id, data_utils.id_to_words, data_utils.id_to_logic))[0]

    #First, train with held-out data to find number of iterations
    with tf.Session() as sess:
        model, num_steps = train(sess, train_data, validation_data, conf)
        loss, total_acc, sentence_acc = test(sess, test_data, model)
        print("INTERMEDIATE RESULTS:")
        print("  loss         = %0.4f"%loss)
        print("  total_acc    = %0.4f"%total_acc)
        print("  sentence_acc = %0.4f"%sentence_acc)

        #Now train on full data set
        #tf.reset_default_graph()
        #train_data += validation_data
        #with tf.Session() as sess:
        #model, _ = train(sess, train_data, None, conf, num_steps)
        model_path = os.path.join(FLAGS.train_dir, 'final_model')
        model.saver.save(sess, model_path)
        conf_out = open(os.path.join(FLAGS.train_dir, 'final_model.conf'), 'w')
        pickle.dump(conf, conf_out)
        conf_out.close()
        loss, total_acc, sentence_acc = test(sess, test_data, model)
        print("FINAL RESULTS:")
        print("  loss         = %0.4f"%loss)
        print("  total_acc    = %0.4f"%total_acc)
        print("  sentence_acc = %0.4f"%sentence_acc)

def main_test():
    _, test_data, (source_vocab_size, target_vocab_size), _, _ = load_data()
    test_conf_path = os.path.join(FLAGS.train_dir, 'final_model.conf')
    conf_in = open(test_conf_path, 'r')
    conf = pickle.load(conf_in)
    conf_in.close()
    with tf.Session() as sess:
        model = create_model(sess, conf, None)
        model.saver.restore(sess, os.path.join(FLAGS.train_dir, 'final_model'))
        loss, total_acc, sentence_acc = test(sess, test_data, model, True)
        print("FINAL RESULTS:")
        print("  loss         = %0.4f"%loss)
        print("  total_acc    = %0.4f"%total_acc)
        print("  sentence_acc = %0.4f"%sentence_acc)


def main(_):
    if FLAGS.mode == "train":
        main_train()
    else:
        main_test()

if __name__ == "__main__":
    tf.app.run()
