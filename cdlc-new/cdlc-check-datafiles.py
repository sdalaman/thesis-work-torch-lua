
# coding: utf-8


import os
import math
#import torch
#import torch.autograd as autograd
#import torch.nn as nn
#import torch.nn.functional as F
#import torch.nn.init as init
#import torch.optim as optim
import pickle
import datetime
import numpy as np

class args(object):
    pass
    
def getData(data_path,classname):
    pos_cnt = 0
    neg_cnt = 0
    path = data_path+'/'+classname+'/positive'
    if os.path.exists(path):
      for file in os.listdir(path):
        with open(path+"/"+file, 'r') as f:
            text = f.read()
            words = text.split(" ")
            if len(words) > 0:
            	pos_cnt += 1

    path = data_path + '/' + classname + '/negative'
    if os.path.exists(path):
      for file in os.listdir(path):
        with open(path+"/"+file, 'r') as f:
            text = f.read()
            words = text.split(" ")
            if len(words) > 0:
            	neg_cnt += 1
    return pos_cnt,neg_cnt

classesOld = ['art','arts','biology','business','creativity','culture','design','economics','education','entertainment','health','politics','science','technology']
classes = ['art','arts','biology','business','creativity','culture','design','economics','education','entertainment','global','health','politics','science','technology']
#classes = ['art']
data_pathPri = "/home/saban/work/additive/data/cdlc_en_tr/english"
data_pathSec = "/home/saban/work/additive/data/cdlc_en_tr/turkish"

load = False

fResOut = open('cdlc-datafiles-list.txt', "w")
results  = []

results.append("\n\n")
results.append("train old data files stats")
results.append("----------------------")

for classname in classesOld:
    if load == False:
        positivePriCnt, negativePriCnt = getData(data_pathPri,classname)
        positiveSecCnt, negativeSecCnt = getData(data_pathSec,classname)
        res = " class : %s posPriCnt %d negPriCnt %d posSecCnt %d negSecCnt %d " % (classname,positivePriCnt,negativePriCnt,positiveSecCnt,negativeSecCnt)
        results.append(res)

data_pathPri = "/home/saban/work/additive/data/cdlc_en_tr/train/englishTok"
data_pathSec = "/home/saban/work/additive/data/cdlc_en_tr/train/turkishTok"
results.append("\n\n")
results.append("2nd train data files stats")
results.append("----------------------")

for classname in classes:
    if load == False:
        positivePriCnt, negativePriCnt = getData(data_pathPri,classname)
        positiveSecCnt, negativeSecCnt = getData(data_pathSec,classname)
        res = " class : %s posPriCnt %d negPriCnt %d posSecCnt %d negSecCnt %d " % (classname,positivePriCnt,negativePriCnt,positiveSecCnt,negativeSecCnt)
        results.append(res)

data_pathPri = "/home/saban/work/additive/data/cdlc_en_tr/test/englishTok"
data_pathSec = "/home/saban/work/additive/data/cdlc_en_tr/test/turkishTok"

results.append("\n\n")
results.append("test data files stats")
results.append("----------------------")

for classname in classes:
    if load == False:
        positivePriCnt, negativePriCnt = getData(data_pathPri,classname)
        positiveSecCnt, negativeSecCnt = getData(data_pathSec,classname)
        res = " class : %s posPriCnt %d negPriCnt %d posSecCnt %d negSecCnt %d " % (classname,positivePriCnt,negativePriCnt,positiveSecCnt,negativeSecCnt)
        results.append(res)


data_pathPri = "/home/saban/work/additive/data/cdlc_en_tr/test/englishMorph"
data_pathSec = "/home/saban/work/additive/data/cdlc_en_tr/test/turkishMorph"

results.append("\n\n")
results.append("morph test data files stats")
results.append("----------------------")

for classname in classes:
    if load == False:
        positivePriCnt, negativePriCnt = getData(data_pathPri,classname)
        positiveSecCnt, negativeSecCnt = getData(data_pathSec,classname)
        res = " class : %s posPriCnt %d negPriCnt %d posSecCnt %d negSecCnt %d " % (classname,positivePriCnt,negativePriCnt,positiveSecCnt,negativeSecCnt)
        results.append(res)


for res in results:
    print(res)
    fResOut.write(res)
    fResOut.write("\n")

fResOut.close()

print("End -------")

