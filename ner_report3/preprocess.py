#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import glob
from subprocess import Popen, PIPE
from wikiProcess import wikiProcess
import sys
import io
import unicodedata
import re


__author__ = "ArkhipovK"
__copyright__ = "ArkhipovK"
__license__ = "none"

reload(sys)  # Reload does the trick!
sys.setdefaultencoding('utf-8')
    
       
def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFC', input_str)
    return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])

def main():
    parser = argparse.ArgumentParser(description='Text preprocessing tool')

    parser.add_argument('path', nargs='+', help='Path of a file or a folder of files.')
    parser.add_argument('-output', default='text/current/processed.txt', help='Path of a file or a folder of files.')
    parser.add_argument('-e', '--extension', default='', help='File extension to filter by.')
    parser.add_argument('-wikiText', action='store_true', help='Process texts from wikidump')
    parser.add_argument('-toLower', action='store_true', help='Lowercase all words')
    parser.add_argument('-wipeChars', default='[^a-zA-Zа-яА-Яё0-9-— .\n]', help='Regexp for pattern to be wiped')
    
    args = parser.parse_args()

    # Parse paths
    full_paths = [os.path.join(os.getcwd(), path) for path in args.path]
    files = set()
    for path in full_paths:
        if os.path.isfile(path):
            files.add(path)
        else:
            for file in os.listdir(path):
                if os.path.isfile(os.path.join(path, file)):
                    files.add(os.path.join(path, file))

    processedPath = args.output

    if (args.wikiText):
        extractedPath = wikiProcess(files)
        # extractedPath = 'text/current/extracted.txt'
        with io.open(extractedPath, 'r', encoding="utf-8") as text, io.open(processedPath, 'w', encoding="utf-8") as processed:
            for line in text:
                if (line == '\n'):
                    continue
                line = line.decode('utf-8')
                if (args.toLower):
                    line = line.lower()
                
                line = remove_accents(line)
                line = ' '.join(line.split())
                pattern = args.wipeChars.decode('utf-8')
                line = re.sub(pattern, '', line)
                line+="\n"
                line = line.decode('utf-8')

                processed.write(line)
                               
    else:
        return

if __name__ == '__main__':
    main()

