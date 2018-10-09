import os
import shutil
from subprocess import Popen, PIPE
import json

def json_load_byteified(file_handle):
    return _byteify(
        json.load(file_handle, object_hook=_byteify),
        ignore_dicts=True
    )

def json_loads_byteified(json_text):
    return _byteify(
        json.loads(json_text, object_hook=_byteify),
        ignore_dicts=True
    )

def _byteify(data, ignore_dicts = False):
    # if this is a unicode string, return its string representation
    if isinstance(data, unicode):
        return data.encode('utf-8')
    # if this is a list of values, return list of byteified values
    if isinstance(data, list):
        return [ _byteify(item, ignore_dicts=True) for item in data ]
    # if this is a dictionary, return dictionary of byteified keys and values
    # but only if we haven't already byteified it
    if isinstance(data, dict) and not ignore_dicts:
        return {
            _byteify(key, ignore_dicts=True): _byteify(value, ignore_dicts=True)
            for key, value in data.iteritems()
        }
    # if it anything else, return it in its original form
    return data
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
        args = ['WikiExtractor.py']+[wikiFile]+["--output", curRawPath]+["-b", bytesLimit]
        if(json):
            args += ["--json"]

        p = Popen(args, stdin=PIPE, stdout=PIPE, stderr=PIPE)
        output, err = p.communicate()

        if(p.poll()!=0):
            print err
            raise Exception(err)
            
        for filename in os.listdir(curRawPath+"AA/"):
            with open(curRawPath+"AA/"+filename, 'r') as json_string, open(processedPath,'a+') as output:
                partial = ''
                for line in json_string:
                    partial += line.rstrip('\n')
                    # partial = partial.encode("utf-8")
                    try:
                        article = json_loads_byteified(partial)
                        output.write(article["text"])
                        partial = ''
                    except ValueError:
                        continue  # Not yet a complete JSON value
    return processedPath
# wikiProcess(["Corpora/animals/wiki/test/wikiData1.xml","Corpora/animals/wiki/test/wikiData2.xml"])       
       
        
