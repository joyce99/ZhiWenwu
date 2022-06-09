from sentence_transformers import SentenceTransformer, util
# 获得句子的字典定义信息得分
import embeddings


def getSentenceBERTScore(BERTmodel, src, tgt):
    srcVec = BERTmodel.encode(src, convert_to_tensor=True)
    tgtVec = BERTmodel.encode(tgt, convert_to_tensor=True)
    cosinScore = util.cos_sim(srcVec,tgtVec)
    return cosinScore

def getSeedByCLDF(BERTmodel,sourceDefinitionFile, targetDefinitionFile,seedFile):
    seedList = []
    sourceDefinitionFile = open(sourceDefinitionFile,"r", encoding='utf-8')
    targetDefinitionFile = open(targetDefinitionFile,"r",encoding='utf-8')
    # 得到键值对形式，key；word（str）,value: definition（list）
    srcDefinitionDict = embeddings.readDefinitionFile(sourceDefinitionFile)
    tgtDefinitionDict = embeddings.readDefinitionFile(targetDefinitionFile)
    seedFile = open(seedFile,"w",encoding='utf-8')
    for srcWord in srcDefinitionDict.keys():
        srcdefinitionList = srcDefinitionDict.get(srcWord)
        for srcdefinition in srcdefinitionList:
            for tgtWord in tgtDefinitionDict.keys():
                tgtdefinitionList = tgtDefinitionDict.get(tgtWord)
                for tgtdefinition in tgtdefinitionList:
                    # print(srcWord + "->>>>" + tgtWord)
                    score = getSentenceBERTScore(BERTmodel,srcdefinition,tgtdefinition)
                    if(score >= 0.8):
                        print("get Seed : " + srcWord + "->>>>" + tgtWord )
                        seedFile.write(srcWord)
                        seedFile.write("	")
                        seedFile.write(tgtWord)
                        seedFile.write("\n")
    sourceDefinitionFile.close()
    targetDefinitionFile.close()
    seedFile.close()
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
getSeedByCLDF(model,"data/dictionary/post.en.en.txt","data/dictionary/post.es.es.txt","seedFile.en.es.txt")