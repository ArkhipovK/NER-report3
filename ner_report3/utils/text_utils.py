#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import shutil
from subprocess import Popen, PIPE
import json
import io

def cleanDir(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)
def wikiProcess(input, json=True, bytesLimit="100M"):
    rawPath = "text/rawwiki/"
    curRawPath = ""
    processedPath = 'text/current/extracted.txt'
    if(os.path.exists(processedPath)):
        with open(processedPath, 'w'): pass

    cleanDir(rawPath)

    for wikiFile in input:
        
        fDir=os.path.basename(wikiFile)
        fDir=os.path.splitext(fDir)[0]

        curRawPath = rawPath+fDir+"/"
        args = [sys.executable]+['-m']+['WikiExtractor']+[wikiFile]+["--output", curRawPath]+["-b", bytesLimit]
        if(json):
            args += ["--json"]
        p = Popen(args, stdin=PIPE, stdout=PIPE, stderr=PIPE)
        output, err = p.communicate()

        if(p.poll()!=0):
            print(err)
            raise Exception(err)
            
        for filename in os.listdir(curRawPath+"AA/"):
            with open(curRawPath+"AA/"+filename, 'r') as json_string, open(processedPath,'a+') as output:
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

        
