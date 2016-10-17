import sys
from nltk.tokenize import RegexpTokenizer
from nltk import word_tokenize

if len(sys.argv) != 4:
    print "Usage: python convertNLMapsData.py <Sentence file> <Logic File> <Output File>"
    sys.exit(1)

logic_tokenizer = RegexpTokenizer("[\w\:]+|.")

with open(sys.argv[1], 'r') as sentFile, open(sys.argv[2], 'r') as logicFile, open(sys.argv[3], 'w') as fout:
    sentences = sentFile.read().splitlines()
    logics = logicFile.read().splitlines()
    for i in xrange(len(sentences)):
        sent_tokens = word_tokenize(sentences[i])
        logic_tokens = logic_tokenizer.tokenize(logics[i])
        fout.write("<s> %s </s>===<s> %s </s>\n\n"%(" ".join(sent_tokens), " ".join(logic_tokens)))

