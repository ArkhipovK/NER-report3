import os
import sys
import six
import argparse
import io
import _pickle as cPickle
import numpy as np
import pandas as pd
import importlib
import inspect
import keras.layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model, Input, load_model
from keras.layers import Embedding, Dense, TimeDistributed, Dropout, Bidirectional,LSTM
from keras.callbacks import Callback
from keras.initializers import Constant
from keras import regularizers
from keras_contrib.layers import CRF
from keras_contrib.utils import save_load_utils
from sklearn.model_selection import train_test_split
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from ner_report3 import __config__

# import matplotlib.pyplot as plt

def learn(model, embMatrix = '', dim = 20, testSize = 0.1, epoch = "5", batchSize = 40, arch = "LSTM", units=50, activation = "tanh", dropout=0.0, recurrent_dropout=0.1,  kernel_regularizer=None, recurrent_regularizer=None,  bidirectional = True, is_crf = True ):
    
    _MODEL_ = os.path.join(__config__['GLOBAL']['storepath'], "models/keras/", model)
    _EPOCH_ = int(epoch)
    _BATCH_SIZE_ = int(batchSize)
    _TEST_SIZE_ = float(testSize)
    _ARCH_ = getattr(importlib.import_module("keras.layers"), arch)

    _ARGS_ = {
        "units" : int(units),
        "activation" : activation,
        "dropout" : float(dropout),
        "recurrent_dropout": float(recurrent_dropout),
        "return_sequences": True
    }
    if(kernel_regularizer):
        _ARGS_["kernel_regularizer"] = regularizers.l2(float(kernel_regularizer))
    if(recurrent_regularizer):
        _ARGS_["recurrent_regularizer"] = regularizers.l2(float(recurrent_regularizer))
    
    _BI_DIRECTIONAL_ = bool(bidirectional)
    _IS_CRF_ = bool(is_crf)
 
    WORDS_MODEL = os.path.join(__config__['GLOBAL']['storepath'], embMatrix)
    DATAPATH = os.path.join(__config__['GLOBAL']['storepath'], "intermediate")
    EMBEDDING_DIM = int(dim)

    print('Processing text dataset')
 
    word_index = cPickle.load( open(os.path.join(DATAPATH,'word_index.p'), 'rb'))
    label_index = cPickle.load( open(os.path.join(DATAPATH,'label_index.p'), 'rb'))
    maxSeqLen = cPickle.load( open(os.path.join(DATAPATH,'maxSeqLen.p'), 'rb'))
    n_tags = len(label_index)
    
    print('Found %s unique tokens.' % len(word_index))
    print('Number of classes is %s.' % n_tags)
    print('Max len of sequence is %s.' % maxSeqLen)
      
    # prepare embedding matrix (now not used)
    num_words = len(word_index) + 1
    embeddings_index = {}
    # embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    # for word, i in word_index.items():
    #     embedding_vector = embeddings_index.get(word)
    #     if embedding_vector is not None:
    #         # words not found in embedding index will be all-zeros.
    #         embedding_matrix[i] = embedding_vector
    

    
    
    with open(WORDS_MODEL) as f:
        for i, line in enumerate(f):
            if(i==0):
                continue
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    print('Found %s word vectors.' % len(embeddings_index))
    sequences = cPickle.load( open(DATAPATH+'/sequences.p', 'rb'))  

    X = [[word_index[w[0]] for w in s] for s in sequences]
    X = pad_sequences(maxlen=maxSeqLen, sequences=X, padding="post")

    Y = [[label_index[w[1]] for w in s] for s in sequences]
    Y = pad_sequences(maxlen=maxSeqLen, sequences=Y, padding="post")


    Y = [to_categorical(i, num_classes=n_tags) for i in Y]

    
    X_tr, X_te, Y_tr, Y_te = train_test_split(X, Y, test_size=_TEST_SIZE_)

    embedding_layer = Embedding(num_words,
                        EMBEDDING_DIM,
                        # embeddings_initializer=Constant(embedding_matrix),
                        mask_zero=True,
                        input_length=maxSeqLen,
                        ) 
    input = Input(shape=(maxSeqLen,))
    model = embedding_layer(input)

    arch_args = {p.name: p for p in inspect.signature(_ARCH_).parameters.values()}

    _ARGS_ = { k : _ARGS_[k] for k in _ARGS_ if k in arch_args }  
    
    loss = "categorical_crossentropy"
    metrics=['accuracy']  

    if(_BI_DIRECTIONAL_):
        model = Bidirectional(_ARCH_(**_ARGS_))(model)  # variational biLSTM
    else:
        model = _ARCH_(**_ARGS_)(model)

    if(_IS_CRF_):
       
        model = TimeDistributed(Dense(10, activation="relu", kernel_regularizer=regularizers.l2(0.01)))(model)
        crf = CRF(n_tags)  # CRF layer
        out = crf(model)  # output
        loss = crf.loss_function
        metrics=[crf.accuracy]
    else:
        out = TimeDistributed(Dense(n_tags, activation="softmax", kernel_regularizer=regularizers.l2(0.01)))(model)
    
    model = Model(input, out)
    model.compile(optimizer="rmsprop", loss=loss, metrics=metrics)
    history = model.fit(X_tr, np.array(Y_tr), batch_size=_BATCH_SIZE_, epochs=_EPOCH_, validation_split=0.1, verbose=1)
    # history = model.fit(X_tr, np.array(Y_tr), batch_size=_BATCH_SIZE_, epochs=_EPOCH_, validation_split=0.1, verbose=1, callbacks=[WeightsSaver(100, _MODEL_)])
    
    model.summary()
    hist = pd.DataFrame(history.history)

    print(hist)
    model.save(_MODEL_ + ".h5")
    
    modelDir = os.path.basename(_MODEL_)
    modelDir = os.path.splitext(modelDir)[0]
    modelDir = os.path.join(os.path.dirname(os.path.realpath(_MODEL_)),modelDir)

    if not os.path.exists(modelDir):
        os.makedirs(modelDir)

    cPickle.dump(X_te, open(os.path.join(modelDir,'X_te.p'), 'wb+')) 
    cPickle.dump(Y_te, open(os.path.join(modelDir,'Y_te.p'), 'wb+'))
    plot_model(model, to_file=os.path.join(modelDir,'arch.png'), show_shapes=True)

    # plt.style.use("ggplot")
    # plt.figure(figsize=(12,12))
    # plt.plot(hist["viterbi_acc"])
    # plt.plot(hist["val_viterbi_acc"])
    # plt.show()

def main():
    parser = argparse.ArgumentParser(description='Neural network learn tool')

    parser.add_argument('-model', default=__config__['ner_report3.learn']['model'], help='Path to save model (Default = %s)' % __config__['ner_report3.learn']['model'])
    parser.add_argument('-embMatrix', default=__config__['ner_report3.learn']['embmatrix'], help='Path for words representation model (Default = %s)' % __config__['ner_report3.learn']['embmatrix'])
    parser.add_argument('-dim', default=__config__['ner_report3.learn']['dim'], help='Size of word vectors.(Default = %s) ' % __config__['ner_report3.learn']['dim'])
    parser.add_argument('-testSize', default=__config__['ner_report3.learn']['testsize'], help='Slice of the test set in relation to the total sample (Default = %s)' % __config__['ner_report3.learn']['testsize'])
    parser.add_argument('-batchSize', default=__config__['ner_report3.learn']['batchsize'], help='Size of batches for learning (Default = %s)' % __config__['ner_report3.learn']['batchsize'])
    parser.add_argument('-epoch', default=__config__['ner_report3.learn']['epoch'], help='Epochs number (Default = %s)' % __config__['ner_report3.learn']['epoch'])
    parser.add_argument('-arch', default=__config__['ner_report3.learn']['arch'], help='Name of RNN architecture to use. See Keras RNN Layers for details (Default = %s)' % __config__['ner_report3.learn']['arch'])
    parser.add_argument('-units', default=__config__['ner_report3.learn']['units'], help='Number of neurons on the layer (Default = %s)' % __config__['ner_report3.learn']['units'])
    parser.add_argument('-act', default=__config__['ner_report3.learn']['act'], help='Activation function name (Default = %s)' % __config__['ner_report3.learn']['act'])
    parser.add_argument('-drop', default=__config__['ner_report3.learn']['drop'], help='Dropout parameter on the input of recurrent layer (Default = %s)' % __config__['ner_report3.learn']['drop'])
    parser.add_argument('-recdrop', default=__config__['ner_report3.learn']['recdrop'], help='Dropout parameter on recurrent layer state (Default = %s)' % __config__['ner_report3.learn']['recdrop'])
    parser.add_argument('-kernreg', default=__config__['ner_report3.learn']['kernreg'], help='L2 regularizer value on the input of recurrent layer (Default = %s)' % __config__['ner_report3.learn']['kernreg'])
    parser.add_argument('-recreg', default=__config__['ner_report3.learn']['recreg'], help='L2 regularizer value on recurrent layer state (Default = %s)' % __config__['ner_report3.learn']['recreg'])
    parser.add_argument('-bi', default=__config__.getboolean('ner_report3.learn', 'bi'), help='Use BiDirectional RNN? (Default = %s)' % __config__['ner_report3.learn']['bi'])
    parser.add_argument('-crf', default=__config__.getboolean('ner_report3.learn', 'crf'), help='Use CRF on the top of model? (Default = %s)' % __config__['ner_report3.learn']['crf'])


    args = parser.parse_args()

    learn(args.model,  args.embMatrix, args.dim,  args.testSize, args.epoch, args.batchSize, args.arch, args.units, args.act, args.drop, args.recdrop, args.kernreg, args.recreg, args.bi, args.crf)
    

    

if __name__ == '__main__':
    main()
