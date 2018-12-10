import numpy as np
from keras_contrib.layers import CRF

def cat2label(pred, index_label):
    out = []
    for pred_i in pred:
        out_i = []
        for p in pred_i:
            p_i = np.argmax(p)
            out_i.append(str(index_label[p_i]))
        out.append(out_i)
    return out
    
def pred2label(pred, index_label):
    out = []
    pred = np.floor(pred)
    for pred_i in pred:
        out_i = []
        for p in pred_i:
                out_i.append(index_label[int(p)])
        out.append(out_i)
    return out

def create_crf_objects():
    instanceHolder = {"instance": None}
    class ClassWrapper(CRF):
        def __init__(self, *args, **kwargs):
            instanceHolder["instance"] = self
            super(ClassWrapper, self).__init__(*args, **kwargs)
    def loss(*args):
        method = getattr(instanceHolder["instance"], "loss_function")
        return method(*args)
    def accuracy(*args):
        method = getattr(instanceHolder["instance"], "accuracy")
        return method(*args)
    return {"ClassWrapper": ClassWrapper ,"CRF": ClassWrapper, "loss": loss, "accuracy":accuracy}

def nested_set(dic, keys):
    for key in keys[:-1]:
        dic = dic.setdefault(key, {})
    dic[keys[-1]] = {"_complete":0}

# def getLabelsOld(path = "labels/animals.txt", div = "_", toLower = True):

#     path = os.path.join(__config__['GLOBAL']['storepath'], path) 

#     labelsHash = {}
#     stemmer = SnowballStemmer("russian")
#     with io.open(path, 'r', encoding="utf-8") as labels:
#         for label in labels:
#             if(toLower):
#                 label = label.lower()
#             label = re.sub(u'\(.*\)','',label)
#             label = re.sub(u'\n','',label)
#             label = label.split(div)

#             label=[stemmer.stem(word) for word in label]

#             if(len(label) == 1):
#                 if label[0] not in labelsHash:
#                     labelsHash[label[0]] = {"_self":1}
#             else:
#                 for i in label:
#                     for j in label:
#                         if(i not in labelsHash):
#                             labelsHash[i] = {}
#                         if(j != i and j not in labelsHash[i]):
#                             labelsHash[i].update({j:1})
#     return labelsHash

