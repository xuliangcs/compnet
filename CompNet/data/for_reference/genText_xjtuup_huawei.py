import os
import numpy as np

path = '/home/sunny/datasets/XJTU-UP/renamed/huawei/Nature'
# path = '/home/sunny/datasets/XJTU-UP/renamed/huawei/Flash'


root = './'
with open(os.path.join(root, 'train.txt'), 'w') as ofs:
    files = os.listdir(path)
    files.sort()
    for filename in files:
        # print(filename)
        userID = int(filename[:4])-1
        sampleID = int(filename[5:9])
        # print(sampleID)
        if sampleID>3:
            continue
        print(userID)
        imagePath = os.path.join(path, filename)
        ofs.write('%s %d\n'%(imagePath,userID))

with open(os.path.join(root, 'test.txt'), 'w') as ofs:
    files = os.listdir(path)
    files.sort()
    for filename in files:
        # print(filename)
        userID = int(filename[:4])-1
        sampleID = int(filename[5:9])
        # print(sampleID)
        if sampleID<=3:
            continue
        print(userID)
        imagePath = os.path.join(path, filename)
        ofs.write('%s %d\n'%(imagePath,userID))