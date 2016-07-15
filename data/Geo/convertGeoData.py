import os

INPUT_DIR = "./UBL/experiments/geo880-lambda/data/"
OUTPUT_DIR = "./geo/"

IGNORE = ["geosents880-typed.ccg.orig", "geo600.dev.giza_probs", "geosents880-typed.ccg"]
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

for f in os.listdir(INPUT_DIR):
    if f in IGNORE:
        continue
    with open(INPUT_DIR+f, "r") as fin:
        result = []
        sent = None
        formula = None
        for line in fin:
            print line.strip()
            if line.startswith("//"):
                continue
            elif sent == None:
                sent = "<s> "+line.strip()+" </s>"
            elif formula == None:
                formula = []
                for token in line.strip().split():
                    if token[0] == "(":
                        formula.append(token[0])
                        formula.append(token[1:])
                    elif token[-1] == ")":
                        ind = token.find(")")
                        formula.append(token[:ind])
                        formula += [")"]* (len(token)-ind)
                    else:
                        formula.append(token)
                formula[0] = "<s>"
                formula[-1] = "</s>"
                result.append((sent, formula))
            else:
                sent = None
                formula = None
    with open(OUTPUT_DIR+f, "w") as fout:
        for item in result:
            fout.write("%s===%s\n"%(item[0], " ".join(item[1])))
