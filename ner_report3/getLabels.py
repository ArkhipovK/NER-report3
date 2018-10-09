# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import nltk
import sys
import io
import re
from nltk.tokenize import sent_tokenize
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem.snowball import SnowballStemmer

reload(sys)  # Reload does the trick!
sys.setdefaultencoding('utf-8')

cwd = os.path.dirname(os.path.abspath(__file__))+'/nltk_data'
nltk.data.path.append(cwd)
def printitems(dictObj, indent=0):
    p=[]
    p.append('<ul>\n')
    for k,v in dictObj.iteritems():
        if isinstance(v, dict):
            p.append('<li>'+ k.encode("utf-8")+ ':')
            p.append(printitems(v))
            p.append('</li>')
        else:
            p.append('<li>'+ k.encode("utf-8")+ ':'+ '1' + '</li>')
    p.append('</ul>\n')
    return '\n'.join(p)
def getLabels(path = "labels/animals.txt", div = "_"):
    labelsHash = {}
    with io.open(path, 'r', encoding="utf-8") as labels:
        for label in labels:

            label.decode("utf-8")
            label = re.sub(u'\(.*\)','',label)
            # print(label.encode("utf-8"))
            label = label.split(div)

            stemmer = SnowballStemmer("russian")
            label=[stemmer.stem(word) for word in label]

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
    html = printitems(labelsHash)
    with io.open("labels/restest.txt", 'w+', encoding="utf-8") as test:
        test.write(html.decode("utf-8"))
getLabels()
