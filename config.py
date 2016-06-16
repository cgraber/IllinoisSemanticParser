import itertools


dropout_beam = [0.2, 0.3, 0.4, 0.5]
vector_beam = [150, 200, 250]

class Config(object):


    def __init__(self, source_vocab_size, target_vocab_size, dropout, param, fold=None):
        self.max_gradient = 5
        self.batch_size = 20
        self.initialize_width = 0.08
        self.dropout_rate = dropout
        self.layer_size = param
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.directory = "dropout_%d_vector_%d"
        self.fold = fold

    def get_dir(self):
        if self.fold:
            return self.directory + "_%d"%self.fold
        else:
            return self.directory

def config_beam_search(source_vocab_size, target_vocab_size):
    for dropout, param in itertools.product(dropout_beam, vector_beam):
        yield Config(dropout, param)
