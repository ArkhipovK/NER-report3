# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import sys
import io
import re
import argparse
import _pickle as cPickle

import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem.snowball import SnowballStemmer
from ner_report3 import __config__
from .utils.shared_utils import nested_set

cwd = os.path.join(os.path.dirname(os.path.abspath(__file__)),'nltk_data')
nltk.data.path.append(cwd)

def printSequences(ls , toFile = False):
    path = os.path.join(__config__['GLOBAL']['storepath'],'intermediate/sequences.p')

    start = None
    end = None
    if(ls[0] != "None"):
        start = int(ls[0])
    if(ls[1] != "None"):
        end = int(ls[1])
    
    if os.path.exists(path):
        sequences = cPickle.load(open(path, 'rb'))
        if(toFile):
            with io.open(os.path.join(os.path.dirname(path),"sequences.txt"), 'w', encoding="utf-8") as output:
                for i,s in enumerate(sequences[start:end]):
                    output.write(str(i+1) + ". " + str(s) + "\n")
        else:
            [print(i+1,". ", s, "\n") for i,s in enumerate(sequences[start:end])]
        # for s in sequences[ls[0]:ls[1]]:    
        #     for w in s:
        #         print(w[0]+"_("+w[1]+") ")
        #     print("\n")
    else:
        print("File %s not found, you must generate it first." % path)    
    return


def getLabels(path = "labels/animals.txt", div = "_", toLower = True):

    path = os.path.join(__config__['GLOBAL']['storepath'], path) 

    instanceHash = {}
    stemmer = SnowballStemmer("russian")
    with io.open(path, 'r', encoding="utf-8") as instances:
        for instance in instances:
            if(toLower):
                instance = instance.lower()
            instance = re.sub(u'\(.*\)','',instance)
            instance = re.sub(u'\n','',instance)
            instance = instance.split(div)

            instance=[stemmer.stem(word) for word in instance]
            nested_set(instanceHash, instance)          
    return instanceHash

def getSequences(textPath = 'text/current/processed.txt', labelPath="labels", lowerLabels = True, limitLength = 300):

    textPath = os.path.join(__config__['GLOBAL']['storepath'], textPath) 
    labelPath = os.path.join(__config__['GLOBAL']['storepath'], labelPath) 
    labels = {}
    
    if os.path.isfile(labelPath):
        l_Hash = getLabels(labelPath, toLower = lowerLabels)

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
    maxSent = []
    empltyClass = "O"
    fullLen = sum(1 for line in open(textPath))
    with io.open(textPath, 'r', encoding="utf-8") as text:
        for st, line in enumerate(text):
            # line = re.sub(u'—','',line)
            abst = sent_tokenize(line,'russian')
            for sent in abst: 
                sent = TreebankWordTokenizer().tokenize(sent) 
                if len(sent) > 5 and limitLength >= len(sent):
                    for l_Class, l_Hash in labels.items(): 
                        candidate = False
                        curSeqLen = len(sent)
                        j = 0
                        while j < curSeqLen:
                            word = sent[j]
                            words[word] = 1
                            stem_word = stemmer.stem(word)

                            if(stem_word in l_Hash):
                                inst_sequence = l_Hash[stem_word]
                                completeIndex = None
                                bias = 1
                                if("_complete" in inst_sequence):
                                    completeIndex = j
                                
                                while(curSeqLen > j+bias and stemmer.stem(sent[j+bias]) in inst_sequence):
                                    inst_sequence = inst_sequence[stemmer.stem(sent[j+bias])]
                                    if("_complete" in inst_sequence):
                                        completeIndex = j+bias
                                    bias += 1
                                if(completeIndex != None):
                                    while(j <= completeIndex):
                                        word = sent[j]
                                        words[word] = 1
                                        sent[j] = (word,l_Class)
                                        candidate = True
                                        j += 1
                                    j -= 1

                                else:
                                    sent[j] = (word,empltyClass)
                            else:
                                sent[j] = (word,empltyClass)
                            j+=1
                        if(candidate == True):
                            sentences += [sent]
                            if(len(sent)>maxSeqLen):
                                maxSeqLen = curSeqLen                
            
            # [[print(s.encode("utf-8")) for s in o] for o in batch]  
            # sent = 
            # [print(w) for w in sent]
            if(st>0 and (st % 10000 == 0)):
                print("Processed {0} rows of {1} ".format(str(st), fullLen))

    word_index = {w: i + 1 for i, w in enumerate(words)}
    lkeys = list(labels.keys())
    label_index = {t: i+1 for i, t in enumerate(lkeys)}
    label_index["O"] = 0
    return sentences, word_index, label_index, maxSeqLen
   
    # with io.open("labels/restest.txt", 'w+', encoding="utf-8") as test:
    #     [[ test.write(w[0].decode("utf-8")+"  "+w[1].decode("utf-8")+'\n') for w in s] for s in sentences]
    #     # test.write(html.decode("utf-8"))
    # # [[print(s.encode("utf-8")) for s in o] for o in sentences]
# getSequences(labelPath="labels/animals.txt")   
def main():
    parser = argparse.ArgumentParser(description='Learn data generator')
    parser.add_argument('-textData', default=__config__['ner_report3.seqgen']['textdata'], help='Path for word corpus (Default = %s)' % __config__['ner_report3.seqgen']['textdata'])
    parser.add_argument('-labelData', default=__config__['ner_report3.seqgen']['labeldata'], help='Path for folder with files class instances lists (Default = %s)' % __config__['ner_report3.seqgen']['labeldata'])
    parser.add_argument('-lowerLabels', default=__config__.getboolean('ner_report3.seqgen', 'lowerlabels'), help='Lowercase all labels (Default = %s)' % __config__['ner_report3.seqgen']['lowerlabels'])
    parser.add_argument('-ls', '--list',  nargs='*', help='Print the sequences in range Use: -ls <start> <end>')
    args = parser.parse_args()

    if(args.list):
        print(args.list)
        printSequences(args.list)
        return

    print('Generating learn data... You can find them in intermediate folder')

    TEXT_DATA = args.textData
    LABEL_DATA = args.labelData
    LOWER_LABELS = bool(args.lowerLabels)
    
    

    sequences, word_index, label_index, maxSeqLen  = getSequences(TEXT_DATA, LABEL_DATA, LOWER_LABELS)

    if not os.path.exists(os.path.join(__config__['GLOBAL']['storepath'],'intermediate')):
        os.makedirs(os.path.join(__config__['GLOBAL']['storepath'],'intermediate'))

    cPickle.dump(sequences, open(os.path.join(__config__['GLOBAL']['storepath'],'intermediate/sequences.p'), 'wb+'))  
    cPickle.dump(word_index, open(os.path.join(__config__['GLOBAL']['storepath'],'intermediate/word_index.p'), 'wb+')) 
    cPickle.dump(label_index, open(os.path.join(__config__['GLOBAL']['storepath'],'intermediate/label_index.p'), 'wb+')) 
    cPickle.dump(maxSeqLen, open(os.path.join(__config__['GLOBAL']['storepath'],'intermediate/maxSeqLen.p'), 'wb+'))

if __name__ == '__main__':
    main()