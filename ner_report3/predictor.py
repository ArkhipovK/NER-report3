import os
import sys
import six
import argparse
import re
import _pickle as cPickle
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Model, Input, load_model
from keras_contrib.layers import CRF
from keras_contrib.utils import save_load_utils
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
import nltk
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize import sent_tokenize

from ner_report3 import __config__
from .utils.shared_utils import cat2label, create_crf_objects

cwd = os.path.join(os.path.dirname(os.path.abspath(__file__)),'nltk_data')
print (cwd)
nltk.data.path.append(cwd)

def predict(model, s, l=True, wipeChars='[^a-zA-Zа-яА-Яё0-9-— .\n]', is_crf = False):
    _MODEL_ = os.path.join(__config__['GLOBAL']['storepath'], "models/keras/", model + ".h5")
    # WORDS_MODEL = args.wordsModel
    DATAPATH = os.path.join(__config__['GLOBAL']['storepath'], 'intermediate')
    # EMBEDDING_DIM = int(args.dim)
    INPUT_STRING = s
    TO_LOWER = l
    PATTERN = wipeChars

    # second, prepare text samples and their labels


    word_index = cPickle.load( open(os.path.join(DATAPATH,'word_index.p'), 'rb') )
    label_index = cPickle.load( open(os.path.join(DATAPATH,'label_index.p'), 'rb') )
    maxSeqLen = cPickle.load( open(os.path.join(DATAPATH,'maxSeqLen.p'), 'rb') )
   

    index_word = {i: w for w, i in word_index.items()}
    index_word[0] = "0"
    index_label = {i: w for w, i in label_index.items()}
    index_label[0] = "0"

    print('Found %s unique tokens.' % len(word_index))
    print('Max len of sequence is %s.' % maxSeqLen)
    
    


    if(is_crf):
        model =load_model(_MODEL_, custom_objects=create_crf_objects())
    else:
        model =load_model(_MODEL_)
  
  
    inputStr = INPUT_STRING
    inputStr = re.sub(PATTERN, '', inputStr)
    if(TO_LOWER):
        inputStr = INPUT_STRING.lower()
    inputStr = sent_tokenize(inputStr,'russian')
    sequences = [TreebankWordTokenizer().tokenize(sent) for sent in inputStr if len(sent)>5]        
    X = [[word_index[w] for w in s] for s in sequences]
    X = pad_sequences(maxlen=maxSeqLen, sequences=X, padding="post")
    
    model_pred = model.predict(X, verbose=1)
    label_pred = cat2label(model_pred, index_label)
    print("{:15}||{:5}||{}".format("Word", "Label", "Prediction"))
    print(40 * "=")
    pred_list = model_pred.tolist()
    for s, l, pred in zip(sequences, label_pred, pred_list):
        for s_i, l_i, pred_i in zip(s, l, pred):
            print("{:15}||{:10}||{}".format(s_i, l_i, pred_i))
        print(30 * "-")


def main():
    parser = argparse.ArgumentParser(description='Use your model for predictions')

    parser.add_argument('-model', default=__config__['ner_report3.predict']['model'], help='Path to save model')
    parser.add_argument('-s', default=__config__['ner_report3.predict']['s'], help='String to predict')
    parser.add_argument('-l', action='store_true', help='Lowercase all words')
    parser.add_argument('-wipeChars', default=__config__['ner_report3.predict']['wipechars'], help='Regexp for pattern to be wiped (Default = %s)' % __config__['ner_report3.predict']['wipechars'])
    parser.add_argument('-crf', default=__config__.getboolean('ner_report3.predict', 'crf'), help='Use CRF on the top of model? (Default = %s)' % __config__['ner_report3.predict']['crf'])

    args = parser.parse_args()
    predict(args.model,args.s,args.l,args.wipeChars, args.crf)
    

if __name__ == '__main__':
    main()
