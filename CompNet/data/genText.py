import os
import numpy as np

path1 = '/home/sunny/datasets/Tongji/palmprint/ROI/session1/'
path2 = '/home/sunny/datasets/Tongji/palmprint/ROI/session2/'


root = './'
with open(os.path.join(root, 'train.txt'), 'w') as ofs:
    files = os.listdir(path1)
    files.sort()           
    for filename in files:
        # print(filename)
        userID = int(filename[:5])
        userID = int((userID-1)/10)
        # print(userID)
        imagePath = os.path.join(path1, filename)
        ofs.write('%s %d\n'%(imagePath, userID))

with open(os.path.join(root, 'test.txt'), 'w') as ofs:
    files = os.listdir(path1)
    files.sort()
    for filename in files:
        # print(filename)
        userID = int(filename[:5])
        userID = int((userID-1)/10)
        # print(userID)
        imagePath = os.path.join(path2, filename)
        ofs.write('%s %d\n'%(imagePath,userID))