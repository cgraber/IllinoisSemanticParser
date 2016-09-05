import itertools


dropout_beam = [0.2, 0.3, 0.4, 0.5]
vector_beam = [150, 200, 250]

class Config(object):


    def __init__(self, source_vocab_size, target_vocab_size, num_layers, dropout, layer_size, batch_size, 
                 learning_rate, learning_rate_decay_factor, source_max_len, target_max_len, 
                 words_to_id, id_to_words, id_to_logic, fold=None):
        self.max_gradient = 5
        self.batch_size = batch_size
        self.initialize_width = 0.08
        self.dropout_rate = dropout
        self.layer_size = layer_size
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.num_layers = num_layers
        self.directory = "dropout_%.1f_vector_%d"%(dropout, layer_size)
        self.learning_rate = learning_rate
        self.learning_rate_decay_factor = learning_rate_decay_factor
        self.source_max_len = source_max_len
        self.target_max_len = target_max_len
        self.fold = fold
        self.words_to_id = words_to_id
        self.id_to_words = id_to_words
        self.id_to_logic = id_to_logic

    def get_dir(self):
        if self.fold:
            return self.directory + "_%d/"%self.fold
        else:
            return self.directory+"/"

def config_beam_search(source_vocab_size, target_vocab_size, num_layers, batch_size, 
                       learning_rate, learning_rate_decay_factor, source_max_len, target_max_len,
                       words_to_id, id_to_words, id_to_logic):
    for dropout, layer_size in itertools.product(dropout_beam, vector_beam):
        yield Config(source_vocab_size, target_vocab_size, num_layers, dropout, layer_size, batch_size, 
                     learning_rate, learning_rate_decay_factor, source_max_len, target_max_len,
                     words_to_id, id_to_words, id_to_logic)
