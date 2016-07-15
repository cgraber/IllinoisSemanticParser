import os, collections
from nltk.stem.snowball import SnowballStemmer

PAD = "<PAD>"
PAD_ID = 0
EOS = "</s>"
LOGIC_EOS_ID = None
stemmer = SnowballStemmer("english")
id_to_logic = None

def _read_words(filename):
    with open(filename, "r") as fin:
        result = [[info for info in line.strip().split("===")] for line in fin]
    for i in xrange(len(result)):
        result[i][0] = result[i][0].split()
        result[i][1] = result[i][1].split()
    
    # Preprocessing steps
    for i in xrange(len(result)):
        for j in xrange(len(result[i][0])):
            result[i][0][j] = stemmer.stem(result[i][0][j])
    return result

def _build_vocab(train_path, test_path):
    data = _read_words(train_path) + _read_words(test_path)
    words = sum([[word for word in entry[0]] for entry in data], [])
    words_counter = collections.Counter(words)
    words_to_id = {word:ind+1 for ind,word in enumerate(words_counter.keys())}
    words_to_id[PAD] = PAD_ID

    logic_tokens = sum([[token for token in entry[1]] for entry in data], [])
    logic_counter = collections.Counter(logic_tokens)
    logic_to_id = {word:ind+1 for ind,word in enumerate(logic_counter.keys())}
    logic_to_id[PAD] = PAD_ID
    global id_to_logic, LOGIC_EOS_ID
    id_to_logic = [None]*len(logic_to_id)
    for key in logic_to_id:
        id_to_logic[logic_to_id[key]] = key
    LOGIC_EOS_ID = logic_to_id[EOS]
    return words_to_id,logic_to_id
    
def _file_to_ids(filename, token_to_id):
    data = _read_words(filename)
    word_ids = [[token_to_id[0][word] for word in entry[0]] for entry in data]
    logic_ids = [[token_to_id[1][logic] for logic in entry[1]] for entry in data]
    return zip(word_ids, logic_ids)

def load_raw_text(path):
    """
    Loads data from directory provided by argument "path"

    It is assumed that this path contains two files: "train.txt" and "test.txt"
    each file is assumed to have format [SENTENCE]===[LOGICAL FORM],
    where the sentences/logical form tokens are separated by a single space.

    """

    train_path = os.path.join(path, "train.txt")
    test_path = os.path.join(path, "test.txt")

    word_to_id = _build_vocab(train_path, test_path)
    train_data = _file_to_ids(train_path, word_to_id)
    test_data = _file_to_ids(test_path, word_to_id)
    vocab_size = (len(word_to_id[0]), len(word_to_id[1]))
    return train_data, test_data, vocab_size

def ids_to_logics(id_list):
    global id_to_logic
    return map(lambda x: id_to_logic[x], id_list)

if __name__=="__main__":
    train, test, vocab = load_raw_text("./data/Geo/")


