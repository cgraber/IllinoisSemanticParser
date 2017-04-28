import sys, random
from nltk import word_tokenize

if len(sys.argv) != 4:
    print "Usage: python convertBlocks.py <input file 1> <input file 2> <output file>"
    sys.exit(1)

result = []
for fname in [sys.argv[1], sys.argv[2]]:
    with open(fname, "r") as fin:
        current = []
        result.append(current)
        sent = None
        formula = None
        config = None
        tempConfig = []
        for line in fin:
            if not line.strip():
                current = []
                result.append(current)
                config = None
            elif '===' in line:
                continue
            elif not formula:
                formula = word_tokenize(line.strip())
                formula.insert(0, "<s>")
                formula.append("</s>")
            else:
                sent = word_tokenize(line.strip())
                sent.insert(0, "<s>")
                sent[-1] = "</s>" # Note that we replace the end-of-sentence full stop with this
                current.append((config, formula, sent))
                sent = None
                formula = None
random.shuffle(result)
with open(sys.argv[3], "w") as fout:
    for current in result:
        for config, formula, sent in current:
            fout.write("%s===%s\n"%(" ".join(sent), " ".join(formula)))
        fout.write("\n")
        

