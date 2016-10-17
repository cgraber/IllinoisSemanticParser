import os, collections, sys
from nltk.stem.snowball import SnowballStemmer

PAD = "<PAD>"
PAD_ID = 0
EOS = "</s>"
stemmer = SnowballStemmer("english")
logic_to_id = None
id_to_logic = None
words_to_id = None
id_to_words = None

def _read_words(filename):
    with open(filename, "r") as fin:
        result = []
        current = None
        for line in fin:
            if not line.strip():
                current = None
            else:
                if current == None:
                    current = []
                    result.append(current)
                info = line.strip().split("===")
                current.append((info[0].split(), info[1].split()))

    # Preprocessing steps
    for i in xrange(len(result)):
        for j in xrange(len(result[i])):
            for k in xrange(len(result[i][j][0])):
                try:
                    result[i][j][0][k] = stemmer.stem(result[i][j][0][k]).lower()
                except:
                    result[i][j][0][k] = result[i][j][0][k].lower()
    return result

def _build_vocab(train_path, test_path):
    data = _read_words(train_path) + _read_words(test_path)
    words = sum(sum([[[word for word in entry[0]] for entry in part] for part in data], []), [])
    words_counter = collections.Counter(words)
    words_to_id = {word:ind+1 for ind,word in enumerate(words_counter.keys())}
    words_to_id[PAD] = PAD_ID

    logic_tokens = sum(sum([[[token for token in entry[1]] for entry in part] for part in data], []), [])
    logic_counter = collections.Counter(logic_tokens)
    logic_to_id = {word:ind+1 for ind,word in enumerate(logic_counter.keys())}
    logic_to_id[PAD] = PAD_ID
    global id_to_logic, LOGIC_EOS_ID, id_to_words, logic_to_id, words_to_id
    id_to_logic = [None]*len(logic_to_id)
    id_to_words = [None]*len(words_to_id)
    for key in logic_to_id:
        id_to_logic[logic_to_id[key]] = key
    for key in words_to_id:
        id_to_words[words_to_id[key]] = key
    LOGIC_EOS_ID = logic_to_id[EOS]
    return words_to_id,logic_to_id
    
def _file_to_ids(filename, token_to_id):
    data = _read_words(filename)
    for i in xrange(len(data)):
        for j in xrange(len(data[i])):
            for k in xrange(len(data[i][j][0])):
                data[i][j][0][k] = token_to_id[0][data[i][j][0][k]]
            for k in xrange(len(data[i][j][1])):
                data[i][j][1][k] = token_to_id[1][data[i][j][1][k]]
    return data

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

def ids_to_words(word_list):
    global id_to_words
    return map(lambda x: id_to_words[x], word_list)

if __name__=="__main__":
    train, test, vocab = load_raw_text("./data/BlocksWorld/", True)
    print test


