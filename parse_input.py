from __future__ import print_function
import tensorflow as tf
import pickle, sys
import config, data_utils, parser_model, pointer_parser_model

if len(sys.argv) != 5:
    print("Usage: python parse_input.py <model file> <config file> <output file> \"[INPUT STRING]\"")
    sys.exit(1)

test_model_path, test_conf_path, output_path = sys.argv[1:4]
conf_in = open(test_conf_path, 'r')
conf = pickle.load(conf_in)
conf_in.close()

with tf.Session() as sess:
    model = pointer_parser_model.MultiSentPointerParseModel(conf, None)
    sess.run(tf.initialize_all_variables())
    model.saver.restore(sess, test_model_path)
    itWorked, result = model.parse(sess, sys.argv[4])

if itWorked:
    result = map(lambda x:"".join(x), result[0])
    result = "^".join(result)
    with open(output_path, "w") as fout:
        fout.write("0\n")
        fout.write(result)

else:
    with open(output_path, "w") as fout:
        fout.write("1\n")
        fout.write("\n".join(result))



