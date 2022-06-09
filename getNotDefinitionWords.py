import re
import time

import embeddings

def getNotDefinitionWords(seedFile, definitionFile ,wordsFile):
    seedFile = open(seedFile,"r",encoding='utf-8')
    definitionFile = open(definitionFile,"r",encoding='utf-8')
    definitionDict = embeddings.readDefinitionFile(definitionFile)
    wordsFile = open(wordsFile,"w",encoding='utf-8')
    wordList = []
    for line in seedFile:
        englishWord, word = line.split(" ")
        if(englishWord.rstrip() not in definitionDict.keys()):
            wordList.append(englishWord.rstrip())
    wordSet = set(wordList)
    for word in wordSet:
        wordsFile.write(word)
        wordsFile.write("\n")
    seedFile.close()
    wordsFile.close()
# getNotDefinitionWords("./data/dictionaries/en-fr.5000-6500.txt","./data/dictionary/all.en.en.txt","./data/dictionary/Seed/en.txt")

def getEnglishDefinitionFromAnotherlanguage(wordsFile,definitionFile, resultFile):
    wordsFile = open(wordsFile,"r",encoding='utf-8')
    definitionFile = open(definitionFile,"r",encoding='utf-8')
    resultFile = open(resultFile,"w",encoding='utf-8')
    definitionDict = embeddings.readDefinitionFile(definitionFile)
    for line in wordsFile:
        word = line.rstrip()
        if(word in definitionDict.keys()):
            definirionList = definitionDict.get(word)
            for definition in definirionList:
                resultFile.write(word)
                resultFile.write("   ")
                pattern = re.compile(r'(\([^\)]*\))')
                definition = re.sub(pattern,"",definition)
                resultFile.write(definition.rstrip())
                resultFile.write("\n")
    wordsFile.close()
    definitionFile.close()
    resultFile.close()
# getEnglishDefinitionFromAnotherlanguage("data/dictionary/Seed/en.txt", "./data/dictionary/all.fr.fr.txt","data/dictionary/Seed/definition-en.txt")



import goslate



def getTranslation(file1,resultFile):
    file = open(file1,"r",encoding='utf-8')
    resultFile = open(resultFile,"w",encoding='utf-8')
    gs = goslate.Goslate()
    for line in file:
        word, definition = line.split("   ",1)
        tran = (gs.translate(definition.rstrip(), 'en'))
        resultFile.write(word)
        resultFile.write("   ")
        resultFile.write(tran)
        resultFile.write("\n")
getTranslation("data/dictionary/Seed/definition-en.txt","data/dictionary/Seed/definition-en.txt1")