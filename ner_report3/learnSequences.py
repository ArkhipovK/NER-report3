from __future__ import print_function

import os
import sys
import six
import argparse
import io
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras_contrib.layers import CRF
from keras.models import Model
from keras.initializers import Constant
from sklearn.model_selection import train_test_split

# import matplotlib.pyplot as plt

from getSequences import getSequences


def main():
    parser = argparse.ArgumentParser(description='Text preprocessing tool')

    parser.add_argument('output', default='models/keras/animals.h5', help='Path to save model')
    parser.add_argument('-epoch', default=100, help='Epochs number')
    parser.add_argument('-batchSize', default=40, help='Size of batches for learning')
    parser.add_argument('-arch', default='BiLSTM', help='Name of architecture to use')
    parser.add_argument('-wordsModel', default='models/fasttext/animals.vec', help='Path for words representation model.')
    parser.add_argument('-textData', default='text/current/processed.txt', help='Path for word corpus.')
    parser.add_argument('-labelData', default='labels', help='Path for folder with files class instances lists')
    parser.add_argument('-dim', default=100, help='Size of word vectors.')
    parser.add_argument('-tokenizer', default='sentence', help='Tokenizer type (sentence or abstract based)')

    args = parser.parse_args()

    _OUTPUT_ = args.output
    _EPOCH_ = args.epoch
    _BATCH_SIZE_ = args.batchSize
    _ARCH_ = args.arch
    
    WORDS_MODEL = args.wordsModel
    TEXT_DATA = args.textData
    LABEL_DATA = args.labelData
    EMBEDDING_DIM = args.dim
    

    embeddings_index = {}

    with open(WORDS_MODEL) as f:
        for i, line in enumerate(f):
            if(i==0):
                continue
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    print('Found %s word vectors.' % len(embeddings_index))

    # second, prepare text samples and their labels
    print('Processing text dataset')

    sequences, word_index, label_index, maxSeqLen  = getSequences(TEXT_DATA, LABEL_DATA)

    print('Found %s unique tokens.' % len(word_index))

                  
    X = [[word_index[w[0]] for w in s] for s in sequences]
    X = pad_sequences(maxlen=maxSeqLen, sequences=X, padding="post")

    Y = [[label_index[w[1]] for w in s] for s in sequences]
    Y = pad_sequences(maxlen=maxSeqLen, sequences=Y, padding="post")

    n_tags = len(label_index.keys())
    Y = [to_categorical(i, num_classes=n_tags) for i in Y]
    
    X_tr, X_te, Y_tr, Y_te = train_test_split(X, Y, test_size=0.1)

    # prepare embedding matrix
    num_words = len(word_index) + 1
    embedding_matrix = np.zeros((num_words, int(EMBEDDING_DIM)))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    
    embedding_layer = Embedding(num_words,
                            int(EMBEDDING_DIM),
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=maxSeqLen,
                            trainable=False) 
    input = Input(shape=(maxSeqLen,))
    model = embedding_layer(input)
    model = Bidirectional(LSTM(units=50, return_sequences=True, recurrent_dropout=0.1))(model)  # variational biLSTM
    model = TimeDistributed(Dense(50, activation="relu"))(model)
    crf = CRF(n_tags)  # CRF layer
    out = crf(model)  # output

    model = Model(input, out)
    model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])

    history = model.fit(X_tr, np.array(Y_tr), batch_size=_BATCH_SIZE_, epochs=_EPOCH_,
                    validation_split=0.1, verbose=1)

    model.summary()

    hist = pd.DataFrame(history.history)

    print(hist)

    # plt.style.use("ggplot")
    # plt.figure(figsize=(12,12))
    # plt.plot(hist["viterbi_acc"])
    # plt.plot(hist["val_viterbi_acc"])
    # plt.show()

    model.save(_OUTPUT_)

if __name__ == '__main__':
    main()
