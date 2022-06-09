from sentence_transformers import SentenceTransformer, util
import embeddings
from cupy_utils import *
import time
import argparse
import collections
import numpy as np
import sys
import datetime
import eval_translation


class WordDefinitionScore(object):
    # BERTmodel:BERT模型， word：字典word2vec词, word2vec：列表, wordDefinition：字典（key:str,value:list）
    def __init__(self, BERTmodel, wordsrc, srcvec, wordtgt, tgtvec, src_word2ind, tgt_word2ind, srcDefinition, tgtDefinition):
        self.BERTmodel = BERTmodel
        self.wordsrc = wordsrc
        self.srcvec = srcvec
        self.wordtgt = wordtgt
        self.tgtvec = tgtvec
        self.src_word2ind = src_word2ind
        self.tgt_word2ind = tgt_word2ind
        self.srcDefinition = srcDefinition
        self.tgtDefinition = tgtDefinition

    # 获得句子的字典定义信息得分
    def getSentenceBERTScore(self, src, tgt):
        srcVec = self.BERTmodel.encode(src, convert_to_tensor=True)
        tgtVec = self.BERTmodel.encode(tgt, convert_to_tensor=True)
        cosinScore = util.cos_sim(srcVec,tgtVec)
        return cosinScore

    # 如果该词的字典定义不存在，则使用源词的字典向量。否则，使用字典定义信息得分
    def getHighestScore(self, srcWord, tgtWord):
        if((srcWord not in self.srcDefinition.keys()) or tgtWord not in  self.tgtDefinition.keys()):
            # 通过编辑距离查找其相似词
            src2ind = self.src_word2ind[srcWord]
            tgt2ind = self.tgt_word2ind[tgtWord]
            vec_src = self.srcvec[src2ind]
            vec_tgt = self.tgtvec[tgt2ind]
            cosin = util.dot_score(vec_src,vec_tgt)
            # if( tgtWord not in  self.tgtDefinition.keys() ): # 如果目标词不在目标语言字典中
            #     print(tgtWord)
            # if( srcWord not in  self.srcDefinition.keys()): # 如果源词不在字典中
            #     print(srcWord)

            return cosin
        #字典都包含源词与目标词的字典定义信息
        srcDefinitionList = self.srcDefinition[srcWord]
        tgtDefinitionList = self.tgtDefinition[tgtWord]
        scores = []
        for i in range(len(srcDefinitionList)):
            for j in range(len(tgtDefinitionList)):
                scores.append(self.getSentenceBERTScore(srcDefinitionList[i],tgtDefinitionList[j]))
        scores.sort(reverse = True)
        return scores
    def getPrecisionByFile(self, dictionaryFile, testFile, Precision):
        split = " "
        dictionaryFile = open(dictionaryFile,"r",encoding='utf-8')
        testFile = open(testFile,"r",encoding='utf-8')
        dictionary = {}
        for line in dictionaryFile:
            word, translationWord = line.split(split,1)
            if(word.rstrip() not in dictionary.keys()):
                list = []
            else:
                list = dictionary[word.rstrip()]
            list.append(translationWord.rstrip())
            dictionary[word] = list
        sumWord = 0
        hintWord = 0
        for line in testFile:
            word, cand = line.split("	",1)
            sumWord = sumWord + 1
            candList = dictionary[word.rstrip()]
            cand = cand.split(" ")
            flag = False
            for i in range(Precision):
                if(flag):
                    break
                for tran in candList:
                    if(cand[i].rstrip()  == tran):
                        hintWord = hintWord + 1
                        flag = True
                        break
        if(sumWord == 0):
            return 0.0
        testFile.close()
        dictionaryFile.close()
        # print(hintWord)
        return float(float(hintWord)/float(sumWord))

    def computeScoreSort(self,sortFile, resultFile, a, b):
        sortFile = open(sortFile, "r", encoding='utf-8')
        result = open(resultFile, "w", encoding='utf-8')
        for line in sortFile:
            srcword, srcs = line.split("	", 1)
            result.write(srcword)
            result.write("	")
            key_value = {}
            srcs = srcs.split(" ")
            step = int((len(srcs) - 1) / 3)
            for i in range(step - 1):
                if (srcs[i * 3].strip() not in key_value.items()):
                    key_value[srcs[i * 3].strip()] = float(srcs[i * 3 + 1]) *a + float(srcs[i * 3 + 2]) * b
            sortList = sorted(key_value.items(), key=lambda kv: kv[1], reverse=True)
            for i in sortList:
                result.write(i[0])
                result.write(" ")
            result.write("\n")
        result.close()
        sortFile.close()
    def filter(self,dictionaty):
        result_dico = []
        for key in dictionaty:
            src_word = key[0]
            tgt_word = key[1]
            if(self.getHighestScore(src_word,tgt_word) >= 0.5):
                result_dico.append(key)
        return result_dico



def eval():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Evaluate embeddings of two languages in a shared space in word translation induction')
    parser.add_argument('src_embeddings', help='the source language embeddings')
    parser.add_argument('trg_embeddings', help='the target language embeddings')
    parser.add_argument('src_definition', help='the source language embeddings')
    parser.add_argument('trg_definition', help='the target language embeddings')
    parser.add_argument('mid_file', help='the mid file')
    parser.add_argument('result_file', help='the result file')
    parser.add_argument('-d', '--dictionary', default=sys.stdin.fileno(),
                        help='the test dictionary file (defaults to stdin)')
    parser.add_argument('--retrieval', default='csls', choices=['nn', 'invnn', 'invsoftmax', 'csls', 'nn1'],
                        help='the retrieval method (nn: standard nearest neighbor; invnn: inverted nearest neighbor; invsoftmax: inverted softmax; csls: cross-domain similarity local scaling)')
    parser.add_argument('--inv_temperature', default=1, type=float,
                        help='the inverse temperature (only compatible with inverted softmax)')
    parser.add_argument('--inv_sample', default=None, type=int,
                        help='use a random subset of the source vocabulary for the inverse computations (only compatible with inverted softmax)')
    parser.add_argument('-k', '--neighborhood', default=10, type=int,
                        help='the neighborhood size (only compatible with csls)')
    parser.add_argument('--dot', action='store_true',
                        help='use the dot product in the similarity computations instead of the cosine')
    parser.add_argument('--encoding', default='utf-8',
                        help='the character encoding for input/output (defaults to utf-8)')
    parser.add_argument('--seed', type=int, default=0, help='the random seed')
    parser.add_argument('--precision', choices=['fp16', 'fp32', 'fp64'], default='fp32',
                        help='the floating-point precision (defaults to fp32)')
    parser.add_argument('--cuda', action='store_true', help='use cuda (requires cupy)')
    args = parser.parse_args()

    # Choose the right dtype for the desired precision
    if args.precision == 'fp16':
        dtype = 'float16'
    elif args.precision == 'fp32':
        dtype = 'float32'
    elif args.precision == 'fp64':
        dtype = 'float64'

    # BERTmodel
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    print("==========load multilingual BERT end ===========")

    # Read input embeddings and words
    srcfile = open(args.src_embeddings, encoding=args.encoding, errors='surrogateescape')
    trgfile = open(args.trg_embeddings, encoding=args.encoding, errors='surrogateescape')
    # wordsrc, srcvec, wordtgt, tgtvec,
    src_words, x = embeddings.read(srcfile, dtype=dtype)
    trg_words, z = embeddings.read(trgfile, dtype=dtype)
    # NumPy/CuPy management
    if args.cuda:
        if not supports_cupy():
            print('ERROR: Install CuPy for CUDA support', file=sys.stderr)
            sys.exit(-1)
        xp = get_cupy()
        x = xp.asarray(x)
        z = xp.asarray(z)
    else:
        xp = np
    xp.random.seed(args.seed)

    # Length normalize embeddings so their dot product effectively computes the cosine similarity
    if not args.dot:
        embeddings.length_normalize(x)
        embeddings.length_normalize(z)

    # Build word to index map,将原来的列表变成键值形式<key,value>，<word,num>'chickweed': 104501, 'bressingham': 104502, 'lonicera': 104503,
    src_word2ind = {word: i for i, word in enumerate(src_words)}
    # 'chickweed': 104501, 'bressingham': 104502, 'lonicera': 104503,
    # print("src_word2ind",src_word2ind)
    trg_word2ind = {word: i for i, word in enumerate(trg_words)}
    print("==========load word embeddings end ===========")
    #打开字典文件
    srcDefinitionfile = open(args.src_definition, encoding=args.encoding, errors='surrogateescape')
    trgDefinitionfile = open(args.trg_definition, encoding=args.encoding, errors='surrogateescape')
    #得到键值对形式，key；word（str）,value: definition（list）
    srcDefinitionDict = embeddings.readDefinitionFile(srcDefinitionfile)
    tgtDefinitionDict = embeddings.readDefinitionFile(trgDefinitionfile)
    print("==========load word definition end ===========")
    WDS = WordDefinitionScore(model, src_words, x, trg_words, z, src_word2ind, trg_word2ind, srcDefinitionDict,tgtDefinitionDict)
    postFile = open(args.mid_file,"r",encoding='utf-8')
    resultFile = open(args.result_file, "w", encoding='utf-8')
    for line in postFile:
        srcword, tgtwords = line.split("	",1)
        tgtwords = tgtwords.split(" ")
        resultFile.write(srcword)
        resultFile.write("	")
        for i in range(len(tgtwords)-1):
            if(i%2 != 0):
                resultFile.write(tgtwords[i].rstrip())
                resultFile.write(" ")
            else:
                resultFile.write(tgtwords[i])
                resultFile.write(" ")
                scores = WDS.getHighestScore(srcword.rstrip(),tgtwords[i].rstrip())
                resultFile.write(str(scores[0].item()))
                resultFile.write(" ")
        resultFile.write("\n")
    resultFile.close()
    postFile.close()

    # def computeScoreAndTestPrecision(self, sortFile, resultFile, dictionaryFile, a, b):


    paramScoreList = [1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.0]
    for a in paramScoreList:
        WDS.computeScoreSort(args.result_file,"final.en-es.txt",a,1.0-a)
        for j in [1,2,3,4,5,6,7,8,9,10] :
            p = WDS.getPrecisionByFile(args.dictionary, "final.en-es.txt", j)
            print("a=" + str(a) + "  b=" + str(1.0-a)+"=============P@" + str(j) + ":"+ str(p))
#   排序

    # print(WDS.getPrecisionByFile(args.dictionary,"final.en-es.txt",1))

if __name__ == '__main__':
    # eval_translation.main()
    print("==========find nn the word end ===========")
    eval()







