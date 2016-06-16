fin = open('np-fixedlex.geo', "r")

result = set()
for line in fin:
    info = line.split(" : ")[1].strip()
    result.add(info)
fin.close()

fout = open('arguments.txt', "w")
for ent in result:
    fout.write("%s\n"%ent)
fout.close()
