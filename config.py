import itertools


dropout_beam = [0.2, 0.3, 0.4, 0.5]
vector_beam = [150, 200, 250]

class Config(object):


    def __init__(self, source_vocab_size, target_vocab_size, num_layers, dropout, layer_size, batch_size, 
                 buckets, learning_rate, learning_rate_decay_factor, num_gpus, fold=None):
        self.max_gradient = 5
        self.batch_size = batch_size
        self.initialize_width = 0.08
        self.dropout_rate = dropout
        self.layer_size = layer_size
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.num_layers = num_layers
        self.directory = "dropout_%.1f_vector_%d"%(dropout, layer_size)
        self.buckets = buckets
        self.learning_rate = learning_rate
        self.learning_rate_decay_factor = learning_rate_decay_factor
        self.fold = fold
        self.num_gpus = num_gpus

    def get_dir(self):
        if self.fold:
            return self.directory + "_%d/"%self.fold
        else:
            return self.directory+"/"

def config_beam_search(source_vocab_size, target_vocab_size, num_layers, batch_size, buckets, 
                       learning_rate, learning_rate_decay_factor, num_gpus):
    for dropout, layer_size in itertools.product(dropout_beam, vector_beam):
        yield Config(source_vocab_size, target_vocab_size, num_layers, dropout, layer_size, batch_size, buckets, 
                     learning_rate, learning_rate_decay_factor, num_gpus)
