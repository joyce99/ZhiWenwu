input = open("wiki.de.vec","r",encoding="utf-8")
out = open("wiki20w.de.vec","w",encoding='utf-8')
i = 0
for line in input:
    out.write(line)
    i = i + 1
    if i==250000:
        break
input.close()
out.close()
print("ads")