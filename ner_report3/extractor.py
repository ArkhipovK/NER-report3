#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import glob
import sys
import io
import unicodedata
import re
import shutil
import json
from subprocess import Popen, PIPE
from ner_report3 import __config__
       
def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFC', input_str)
    return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])
def cleanDir(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)
def wikiProcess(input, json_flag=True, bytesLimit="100M"):
    rawPath = os.path.join(__config__['GLOBAL']['storepath'], 'text/rawwiki/')
    curRawPath = __config__['GLOBAL']['storepath']
    processedPath = os.path.join(__config__['GLOBAL']['storepath'], 'text/current/extracted.txt')
    if(os.path.exists(processedPath)):
        with open(processedPath, 'w'): pass

    cleanDir(rawPath)

    for wikiFile in input:
        
        fDir=os.path.basename(wikiFile)
        fDir=os.path.splitext(fDir)[0]

        curRawPath = rawPath+fDir+"/"
        args = ['WikiExtractor.py']+[wikiFile]+["--output", curRawPath]+["-b", bytesLimit]
        if(json_flag):
            args += ["--json"]
        p = Popen(args, stdin=PIPE, stdout=PIPE, stderr=PIPE)
        output, err = p.communicate()

        if(p.poll()!=0):
            print(err)
            raise Exception(err)
            
        for filename in os.listdir(os.path.join(curRawPath,"AA/")):
            if not os.path.exists(os.path.dirname(processedPath)):
                os.makedirs(os.path.dirname(processedPath))
            with open(os.path.join(curRawPath,"AA/",filename), 'r') as json_string, open(processedPath,'a+') as output:
                partial = ''
                for line in json_string:
                    partial += line.rstrip('\n')
                    try:
                        article = json.loads(partial)
                        output.write(article["text"])
                        partial = ''
                    except ValueError:
                        continue  
    return processedPath
def extract(path, output = 'text/current/', ext = 'txt', wikiText = False, toLower = False, wipeChars = '[^a-zA-Zа-яА-Яё0-9-— .\n]'):
    
    wikiText= bool(wikiText)
    toLower= bool(toLower)
    
    full_paths = [os.path.join(__config__['GLOBAL']['storepath'], p) for p in path]
    files = set()
    for p in full_paths:
        if os.path.isfile(p):
            files.add(p)
        else:
            for file in os.listdir(p):
                if os.path.isfile(os.path.join(p, file)):
                    files.add(os.path.join(p, file))
    processedPath = os.path.join(__config__['GLOBAL']['storepath'],output,'processed.txt')
    if not os.path.exists(os.path.dirname(processedPath)):
        os.makedirs(os.path.dirname(processedPath))
    print('Exctracting files...')
    if (wikiText):
        extractedPath = wikiProcess(files)
        # extractedPath = 'text/current/extracted.txt'
        with io.open(extractedPath, 'r', encoding="utf-8") as text, io.open(processedPath, 'w', encoding="utf-8") as processed:
            for line in text:
                if (line == '\n'):
                    continue
                if (toLower):
                    line = line.lower()
                
                line = remove_accents(line)
                line = ' '.join(line.split())
                line = re.sub(wipeChars, '', line)
                line+="\n"

                processed.write(line)                             
    else:
        return
    print('Files were extracted. Check %s folder' % processedPath)

def main():
    parser = argparse.ArgumentParser(description='Text preprocessing comand')

    parser.add_argument('-path', nargs='+', default= __config__['ner_report3.extract']['path'], help='Path of a file or a folder of files.')
    parser.add_argument('-output', default=__config__['ner_report3.extract']['output'], help='Path of a file or a folder of files.')
    parser.add_argument('-ext', default=__config__['ner_report3.extract']['ext'], help='File extension to filter by.')
    parser.add_argument('-wikiText', action='store_true', help='Process texts from wikidump')
    parser.add_argument('-toLower', action='store_true', help='Lowercase all words')
    parser.add_argument('-wipeChars', default=__config__['ner_report3.extract']['wipechars'], help='Regexp for pattern to be wiped')
    
    args = parser.parse_args()

    

if __name__ == '__main__':
    main()

