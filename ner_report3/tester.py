import os
import sys
import argparse
import io
import _pickle as cPickle
import pandas as pd
import numpy as np
from decimal import Decimal
from keras.models import Model, load_model
from keras_contrib.utils import save_load_utils
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from keras_contrib.layers import CRF
from ner_report3 import __config__
from .utils.shared_utils import cat2label, create_crf_objects



def test(model,is_crf = False): 
    
    _MODEL_ = os.path.join(__config__['GLOBAL']['storepath'], "models/keras/", model + ".h5")

    INDEXPATH = os.path.join(__config__['GLOBAL']['storepath'], "intermediate")

    word_index = cPickle.load( open(os.path.join(INDEXPATH,'word_index.p'), 'rb'))
    label_index = cPickle.load( open(os.path.join(INDEXPATH,'label_index.p'), 'rb'))

    index_word = {i: w for w, i in word_index.items()}
    index_label = {i: w for w, i in label_index.items()}

    modelDir = os.path.basename(_MODEL_)
    modelDir = os.path.splitext(modelDir)[0]
    modelDir = os.path.join(os.path.dirname(os.path.realpath(_MODEL_)),modelDir)

    # save_load_utils.load_all_weights(model, _MODEL_)
    
    if(is_crf):
        model =load_model(_MODEL_, custom_objects=create_crf_objects())
    else:
        model =load_model(_MODEL_)
    X_te = cPickle.load( open(os.path.join(modelDir,'X_te.p'), 'rb')) 
    Y_te = cPickle.load( open(os.path.join(modelDir,'Y_te.p'), 'rb'))
    
    test_pred = model.predict(X_te, verbose=1)

        
    pred_labels = cat2label(test_pred, index_label)
    test_labels = cat2label(Y_te, index_label)

    print("F1-score: {:.1%}".format(f1_score(test_labels, pred_labels)))
    print(classification_report(test_labels, pred_labels))

    # # i = 5
    # for i in X_te:
    #     p = model.predict(np.array([i]))
    #     for w, l in zip(i,p[0]):
    #     # p = np.argmax(p, axis=-1)
    #         if(w!=0):
    #             print("{:5} {}".format(index_word[w], l))
    # true = np.argmax(Y_te[i], -1)
    # print("{:15}||{:5}||{}".format("Word", "True", "Pred"))
    # print(30 * "=")
    # for w, t, pred in zip(X_te[i], true, p[0]):
    #     if w != 0:
    #         print("{:15}: {:5} {}".format(w-1, t, pred)) 
def main():

    parser = argparse.ArgumentParser(description='Neural network test tool')

    parser.add_argument('model', default=__config__['ner_report3.test']['model'], help='Path to model')
    parser.add_argument('-crf', default=__config__.getboolean('ner_report3.test', 'crf'), help='Use CRF on the top of model? (Default = %s)' % __config__['ner_report3.test']['crf'])

    args = parser.parse_args(sys.argv[2:])

    test(args.model, args.crf)

if __name__ == '__main__':
    main()
    