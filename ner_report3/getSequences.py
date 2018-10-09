# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import sys
import io
import re

import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem.snowball import SnowballStemmer

reload(sys)  # Reload does the trick!
sys.setdefaultencoding('utf-8')

cwd = os.path.dirname(os.path.abspath(__file__))+'/nltk_data'
nltk.data.path.append(cwd)

def getLabels(path = "labels/animals.txt", div = "_", toLower = True):
    labelsHash = {}
    stemmer = SnowballStemmer("russian")
    with io.open(path, 'r', encoding="utf-8") as labels:
        for label in labels:

            label = label.decode("utf-8")
            if(toLower):
                label = label.lower()
            label = re.sub(u'\(.*\)','',label)
            label = re.sub(u'\n','',label)
            label = label.split(div)

            label=[stemmer.stem(word.decode("utf-8")) for word in label]

            if(len(label) == 1):
                if label[0] not in labelsHash:
                    labelsHash[label[0]] = {"_self":1}
            else:
                for i in label:
                    for j in label:
                        if(i not in labelsHash):
                            labelsHash[i] = {}
                        if(j != i and j not in labelsHash[i]):
                            labelsHash[i].update({j:1})
    return labelsHash


def getSequences(textPath = 'text/current/processed.txt', labelPath="labels"):

    labels = {}
    
    if os.path.isfile(labelPath):
        l_Hash = getLabels(labelPath)

        l_Class = os.path.basename(labelPath)
        l_Class = os.path.splitext(l_Class)[0]

        labels.update({l_Class:l_Hash})
    else:
        for file in os.listdir(labelPath):
            if os.path.isfile(os.path.join(labelPath, file)):
                labelFile = labelPath+"/"+file
                l_Hash = getLabels(labelFile)

                l_Class = os.path.splitext(file)[0]

                labels.update({l_Class:l_Hash})

    sentences = []
    words = {}
    stemmer = SnowballStemmer("russian")
    maxSeqLen = 0

    with io.open(textPath, 'r', encoding="utf-8") as text:
        for i, line in enumerate(text):
            line = line.decode("utf-8")
            # line = re.sub(u'—','',line)
            abst = sent_tokenize(line.decode("utf-8"),'russian')
            batch = [TreebankWordTokenizer().tokenize(sent) for sent in abst if len(sent)>5]

            for i, sent in enumerate(batch):
                for j, word in enumerate(sent):
                    if(word not in words):
                        words[word]=1
                    curSeqLen = len(sent)
                    if(len(sent)>maxSeqLen):
                        maxSeqLen = curSeqLen

                    stem_word = stemmer.stem(word)
                    actualClass = "_unknown_"
                    # print(word.encode("utf-8")+"  "+stem_word.encode("utf-8"))
                    for l_Class, l_Hash in labels.items():
                        
                        if(stem_word in l_Hash):
                            # print(stem_word.encode("utf-8"))
                            # print(l_Class.encode("utf-8"))
                            # print(l_Hash[stem_word])
                            if('_self' in l_Hash[stem_word]):
                                actualClass = l_Class
                                break
                            elif(curSeqLen > j+1):
                                bias = 1
                                while(curSeqLen > j+bias and stemmer.stem(sent[j+bias]) in l_Hash[stem_word]):
                                    if('_self' in l_Hash[stemmer.stem(sent[j+bias])]):
                                        actualClass = l_Class
                                        break
                                    bias+=1
                                # bias = j-1
                                # while(bias >=0 and stemmer.stem(sent[j-bias]) in l_Hash):
                                #     if('_self' in l_Hash[stemmer.stem(sent[j+bias])]):
                                #         actualClass = l_Class
                                #         break
                                #     bias-=1
                    batch[i][j]= (word,actualClass)
            sentences += [s for s in batch]
            word_index = {w: i + 1 for i, w in enumerate(words)}
            # [[print(s.encode("utf-8")) for s in o] for o in batch]
            lkeys = ["_unknown_"] + labels.keys()
            label_index = {t: i for i, t in enumerate(lkeys)}
            # sent = 
            # [print(w) for w in sent]
            if(i>0 and 100 % i == 0):
                print(i)
            if(i>60):
                print(i)
                break
    
    return sentences, word_index, label_index, maxSeqLen
   
    # with io.open("labels/restest.txt", 'w+', encoding="utf-8") as test:
    #     [[ test.write(w[0].decode("utf-8")+"  "+w[1].decode("utf-8")+'\n') for w in s] for s in sentences]
    #     # test.write(html.decode("utf-8"))
    # # [[print(s.encode("utf-8")) for s in o] for o in sentences]
# getSequences(labelPath="labels/animals.txt")   