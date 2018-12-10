#!/usr/bin/env python
import os
import io
import argparse
import sys
import _pickle as cPickle
from ner_report3 import __config__


__author__ = "ArkhipovK"
__copyright__ = "ArkhipovK"
__license__ = "none"

class Report3(object):

    def __init__(self):
        parser = argparse.ArgumentParser(
            description='Propotype for Named Entity Recognition Module',
            usage='''report3 <command> [<args>]

Available commands are:
    extract    Exctact pure text from your noisy corpus (stage 1 of 5)
    embgen     Custom embedding generator with Word2vec, GloVe and FastText (stage 1.5 of 5 optional)
    seqgen     Convert your texts to train data (stage 2 of 5)
    learn      Learn  the model (stage 3 of 5)
    test      Test the model (stage 4 of 5)
    predict    Use specified model for label predictions (stage 5 of 5)
    config     Edit configuration file
''')
        parser.add_argument('command', help='Subcommand to run')
        # parse_args defaults to [1:] for args, but you need to
        # exclude the rest of the args too, or validation will fail
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()

    def config(self):
        parser = argparse.ArgumentParser(description='Configuration tool')
        parser.add_argument('-ls',  action='store_true', help='Print config')
        parser.add_argument('-sp', help='Path to work directory')
        parser.add_argument("-v", "--value", nargs=3, help='''
        Set a default value for component parameter. 
        E.g -v extract output "text/current" sets "text/current" as a deafault value for output parameter of the component extract
        ''')

        args = parser.parse_args(sys.argv[2:])

        if(args.ls):
            with io.open("config.ini", 'r', encoding="utf-8") as fin:
                print(fin.read(), end="")   
        if(args.sp):    
            __config__['GLOBAL']['storepath'] = args.sp
        if(args.value):    
            __config__[__package__+'.'+args.value[0]][args.value[1]] = args.value[2]

        with open('config.ini', 'w') as configfile:
            __config__.write(configfile)

    def extract(self):
        from .extractor import extract

        parser = argparse.ArgumentParser(description='Text preprocessing comand')

        parser.add_argument('-path', nargs='+', default= __config__['ner_report3.extract']['path'], help='Path of a file or a folder of files (Default = %s)' % __config__['ner_report3.extract']['path'])
        parser.add_argument('-output', default=__config__['ner_report3.extract']['output'], help='Path of a file or a folder of files (Default = %s)' % __config__['ner_report3.extract']['output'])
        parser.add_argument('-ext', default=__config__['ner_report3.extract']['ext'], help='File extension to filter by (Default = *)')
        parser.add_argument('-wikiText', action='store_true', help='Process texts from wikidump')
        parser.add_argument('-toLower', action='store_true', help='Lowercase all words')
        parser.add_argument('-wipeChars', default=__config__['ner_report3.extract']['wipechars'], help='Regexp for pattern to be wiped (Default = %s)' % __config__['ner_report3.extract']['wipechars'])
        
        args = parser.parse_args(sys.argv[2:])

        extract(args.path, args.output, args.ext, args.wikiText, args.toLower, args.wipeChars)
    
    def embgen(self):
        print('Not available in this version')

    def seqgen(self):
        from .seqGenerator import getSequences,printSequences

        parser = argparse.ArgumentParser(description='Learn data generator')
        parser.add_argument('-textData', default=__config__['ner_report3.seqgen']['textdata'], help='Path for word corpus (Default = %s)' % __config__['ner_report3.seqgen']['textdata'])
        parser.add_argument('-labelData', default=__config__['ner_report3.seqgen']['labeldata'], help='Path for folder with files class instances lists (Default = %s)' % __config__['ner_report3.seqgen']['labeldata'])
        parser.add_argument('-lowerLabels', default=__config__.getboolean('ner_report3.seqgen', 'lowerlabels'), help='Lowercase all labels (Default = %s)' % __config__['ner_report3.seqgen']['lowerlabels'])
        parser.add_argument('-lim', default=__config__['ner_report3.seqgen']['lim'], help='Maximum size of sequence (Default = %s)' % __config__['ner_report3.seqgen']['lim'])
        parser.add_argument('-ls', '--list',  nargs='*', help='Print sequences in range Use: -ls <start> <end>')
        parser.add_argument('-dump',  nargs='*', help='Dump sequences in range to txt file Use: -dump <start> <end>')
        args = parser.parse_args(sys.argv[2:])

        if(args.list):
            printSequences(args.list)
            return

        if(args.dump):
            printSequences(args.dump, toFile=True)
            return

        print('Generating learn data... You can find them in intermediate folder')

        TEXT_DATA = args.textData
        LABEL_DATA = args.labelData
        LOWER_LABELS = bool(args.lowerLabels)
        LIM = int(args.lim)
        
        

        sequences, word_index, label_index, maxSeqLen  = getSequences(TEXT_DATA, LABEL_DATA, LOWER_LABELS, LIM)

        if not os.path.exists(os.path.join(__config__['GLOBAL']['storepath'],'intermediate')):
            os.makedirs(os.path.join(__config__['GLOBAL']['storepath'],'intermediate'))

        cPickle.dump(sequences, open(os.path.join(__config__['GLOBAL']['storepath'],'intermediate/sequences.p'), 'wb+'))  
        cPickle.dump(word_index, open(os.path.join(__config__['GLOBAL']['storepath'],'intermediate/word_index.p'), 'wb+')) 
        cPickle.dump(label_index, open(os.path.join(__config__['GLOBAL']['storepath'],'intermediate/label_index.p'), 'wb+')) 
        cPickle.dump(maxSeqLen, open(os.path.join(__config__['GLOBAL']['storepath'],'intermediate/maxSeqLen.p'), 'wb+'))

    def learn(self):
        from .learner import learn

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


        args = parser.parse_args(sys.argv[2:])

        learn(args.model, args.embMatrix, args.dim,  args.testSize, args.epoch, args.batchSize, args.arch, args.units, args.act, args.drop, args.recdrop, args.kernreg, args.recreg, args.bi, args.crf)
    
    def test(self):
        from .tester import test

        parser = argparse.ArgumentParser(description='Neural network test tool')

        parser.add_argument('model', default=__config__['ner_report3.test']['model'], help='Path to model')
        parser.add_argument('-crf', default=__config__.getboolean('ner_report3.test', 'crf'), help='Use CRF on the top of model? (Default = %s)' % __config__['ner_report3.test']['crf'])

        args = parser.parse_args(sys.argv[2:])

        test(args.model, args.crf)
    
    def predict(self):
        from .predictor import predict

        parser = argparse.ArgumentParser(description='Use your model for predictions')

        parser.add_argument('-model', default=__config__['ner_report3.predict']['model'], help='Path to save model')
        parser.add_argument('-s', default=__config__['ner_report3.predict']['s'], help='String to predict')
        parser.add_argument('-l', action='store_true', help='Lowercase all words')
        parser.add_argument('-wipeChars', default=__config__['ner_report3.predict']['wipechars'], help='Regexp for pattern to be wiped (Default = %s)' % __config__['ner_report3.predict']['wipechars'])
        parser.add_argument('-crf', default=__config__.getboolean('ner_report3.predict', 'crf'), help='Use CRF on the top of model? (Default = %s)' % __config__['ner_report3.predict']['crf'])
        args = parser.parse_args(sys.argv[2:])
        predict(args.model,args.s,args.l,args.wipeChars, args.crf)
        

def main():
    
    Report3()

if __name__ == '__main__':
    main()
    
    