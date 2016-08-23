import sys
from nltk import word_tokenize

if len(sys.argv) != 3:
    print "Usage: python convertBlocks.py <input file> <output file>"
    sys.exit(1)

with open(sys.argv[1], "r") as fin:
    result = []
    sent = None
    formula = None
    config = None
    tempConfig = []
    for line in fin:
        if not line.strip():
            continue
        elif line.strip() == "===":
            config = tempConfig
            tempConfig = []
        elif not config:
            tempConfig.append(line.strip().split())
        elif not formula:
            formula = word_tokenize(line.strip())
            formula.insert(0, "<s>")
            formula.append("</s>")
        else:
            sent = word_tokenize(line.strip())
            sent = [x.lower() for x in sent]
            sent.insert(0, "<s>")
            sent[-1] = "</s>" # Note that we replace the end-of-sentence full stop with this
            result.append((config, formula, sent))
            sent = None
            formula = None
            config = None

with open(sys.argv[2], "w") as fout:
    for config, formula, sent in result:
        fout.write("%s===%s\n"%(" ".join(sent), " ".join(formula)))
        

