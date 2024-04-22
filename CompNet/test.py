import torch 
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import transforms
from torchvision import models
import cv2 as cv 
import os
import pickle
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
plt.switch_backend('agg')

from models import MyDataset
from models import compnet
from utils import *



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('\ndevice-> ', device, '\n\n')


path_rst = './rst/veriEER/' 
path_hard = os.path.join(path_rst, 'rank1_hard')


model_path = './net_params.pth'
print('model_path: ', model_path)
print('\n')


train_set_file = './data/train.txt'
test_set_file = './data/test.txt'

trainset =MyDataset(txt=train_set_file, transforms=None, train=False)
testset =MyDataset(txt=test_set_file, transforms=None, train=False)


batch_size = 32#128#8

data_loader_train = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=False)
data_loader_test = DataLoader(dataset=testset, batch_size=batch_size, shuffle=False)



fileDB_train = getFileNames(train_set_file)
fileDB_test = getFileNames(test_set_file)


# output dir
if not os.path.exists(path_rst):
    os.makedirs(path_rst)

if not os.path.exists(path_hard):
    os.makedirs(path_hard)


num_classes = 600  # IITD: 460    KTU: 145    Tongji: 600    REST: 358    XJTU: 200  <---------
net = compnet(num_classes=num_classes)

# print(net)

net.load_state_dict(torch.load(model_path))

net.to(device)
net.eval()


# feature extraction:

featDB_train = []
iddb_train = []

for batch_id, (data, target) in enumerate(data_loader_train):
    
    data = data.to(device)
    target = target.to(device)
    
    codes = net.getFeatureCode(data)
    
    codes = codes.cpu().detach().numpy()
    y = target.cpu().detach().numpy()

    if batch_id == 0:
        featDB_train = codes
        iddb_train =  y
    else:
        featDB_train = np.concatenate((featDB_train, codes), axis=0)
        iddb_train = np.concatenate((iddb_train, y))

print('completed feature extraction for training set.')
print('featDB_train.shape: ', featDB_train.shape)



classNumel = len(set(iddb_train))
num_training_samples = featDB_train.shape[0]
assert num_training_samples % classNumel == 0
trainNum = num_training_samples // classNumel
print('Training DB: ', classNumel, 'classes, ', trainNum, 'images per class')
print('\n')


featDB_test = []
iddb_test = []

for batch_id, (data, target) in enumerate(data_loader_test):
    
    data = data.to(device)
    target = target.to(device)
    
    codes = net.getFeatureCode(data)

    codes = codes.cpu().detach().numpy()
    y = target.cpu().detach().numpy()

    if batch_id == 0:
        featDB_test = codes
        iddb_test =  y
    else:
        featDB_test = np.concatenate((featDB_test, codes), axis=0)
        iddb_test = np.concatenate((iddb_test, y))

print('completed feature extraction for test set.')
print('featDB_test.shape: ', featDB_test.shape)


print('\nfeature extraction done!')
print('\n')
 

print('start feature matching ...')

print('\n\n---------------------')
print('Verification EER of the test set ...')

# verification EER of the test set
s = [] # matching score
l = [] # intra-class or inter-class matching 
ntest = featDB_test.shape[0]
ntrain = featDB_train.shape[0]

for i in range(ntest):
    feat1 = featDB_test[i]

    for j in range(ntrain):        
        feat2 = featDB_train[j]

        
        cosdis =np.dot(feat1,feat2)
        dis = np.arccos(np.clip(cosdis, -1, 1))/np.pi

        s.append(dis)

        if iddb_test[i] == iddb_train[j]: # same palm
            l.append(1)
        else:
            l.append(-1)


with open('./rst/veriEER/scores_VeriEER.txt', 'w') as f:
    for i in range(len(s)):
        score = str(s[i])
        label = str(l[i])
        f.write(score+' '+label+'\n')


os.system('python ./getGI.py   ./rst/veriEER/scores_VeriEER.txt scores_VeriEER')
os.system('python ./getEER.py  ./rst/veriEER/scores_VeriEER.txt scores_VeriEER')  




print('\n-----------')
print('Rank-1 acc of the test set...')
# rank-1 acc
cnt=0
corr=0
for i in range(ntest): 
    probeID = iddb_test[i]

    dis = np.zeros((ntrain, 1))    

    for j in range(ntrain):  
        dis[j] = s[cnt]
        cnt+=1    

    idx = np.argmin(dis[:]) 

    galleryID = iddb_train[idx]

    if probeID == galleryID:
        corr+=1
    else:
        testname = fileDB_test[i]
        trainname = fileDB_train[idx]
        # store similar inter-class samples
        im_test = cv.imread(testname)
        im_train = cv.imread(trainname)
        img = np.concatenate((im_test, im_train), axis=1)
        cv.imwrite('./rst/veriEER/rank1_hard/%6.4f_%s_%s.png'%(np.min(dis[:]), testname[-13:-4], trainname[-13:-4]), img)


rankacc = corr/ntest*100
print('rank-1 acc: %.3f%%'%rankacc)
print('-----------')

with open('./rst/veriEER/rank1.txt', 'w') as f:
    f.write('rank-1 acc: %.3f%%'%rankacc)
       


print('\n\nAggregated verification EER of the test set...')


s = np.array(s)
with open('./rst/veriEER/scores_VeriEER_aggr.txt', 'w') as f:
    for i in range(ntest):
        for j in range(classNumel):
            # print(i*ntrain+j*trainNum, i*ntrain+(j+1)*trainNum, s.shape)
            score = str((s[i*ntrain+j*trainNum : i*ntrain+(j+1)*trainNum]).min())
            label = str(l[i*ntrain+j*trainNum])
            f.write(score+' '+label+'\n')


os.system('python ./getGI.py   ./rst/veriEER/scores_VeriEER_aggr.txt scores_VeriEER_aggr')
os.system('python ./getEER.py  ./rst/veriEER/scores_VeriEER_aggr.txt scores_VeriEER_aggr')    


print('\n\n---------------------')
print('EER of the test set...')
# dataset EER of the test set (the gallery set is not used)
s = [] # matching score
l = [] # genuine / impostor matching
n = featDB_test.shape[0]
for i in range(n-1):
    feat1 = featDB_test[i]

    for jj in range(n-i-1):
        j = i+jj+1        
        feat2 = featDB_test[j]

        cosdis =np.dot(feat1,feat2)
        dis = np.arccos(np.clip(cosdis, -1, 1))/np.pi

        s.append(dis)
        
        if iddb_test[i] == iddb_test[j]:
            l.append(1)
        else:
            l.append(-1)

with open('./rst/veriEER/scores_EER_test.txt', 'w') as f:
    for i in range(len(s)):
        score = str(s[i])
        label = str(l[i])
        f.write(score+' '+label+'\n')

os.system('python ./getGI.py   ./rst/veriEER/scores_EER_test.txt scores_EER_test')
os.system('python ./getEER.py  ./rst/veriEER/scores_EER_test.txt scores_EER_test')
