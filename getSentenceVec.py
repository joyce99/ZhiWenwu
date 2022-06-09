# 将释义句子转化为句子词向量
from sentence_transformers import SentenceTransformer, util
import  time

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
file = open("./data/dictionary/post.en.en.txt","r",encoding='utf-8')
dict = open("./data/dictionary/vec/dict.vec.en","w",encoding='utf-8')
print("start:")
i = 0
for line in file:
    print(i)
    i = i + 1
    str1 = line.split("   ",1)
    if(len(str1) < 2):
        continue
    dict.write(str1[0])
    print(str1[0])
    dict.write("   ")
    arr = (model.encode(str1[1], convert_to_numpy=True))
    vec = ','.join(str(i) for i in arr)
    dict.write(vec)
    dict.write("\n")
print("end:" + time.time())




