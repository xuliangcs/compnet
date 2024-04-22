import torch 
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import DataParallel
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import transforms
from torchvision import models


import time
import sys
import os
import pickle
import numpy as np
from PIL import Image
import cv2


import matplotlib.pyplot as plt
plt.switch_backend('agg')


from models import MyDataset
from models import compnet
from utils import *



# path
train_set_file = './data/train.txt'
test_set_file = './data/test.txt'

path_rst = './rst'


# dataset
trainset = MyDataset(txt=train_set_file, transforms=None, train=True, imside=128, outchannels=1)
testset = MyDataset(txt=test_set_file, transforms=None, train=False, imside=128, outchannels=1)


batch_size = 8


data_loader_train = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
data_loader_test = DataLoader(dataset=testset, batch_size=batch_size, shuffle=False)

data_loader_show = DataLoader(dataset=trainset, batch_size=8, shuffle=True)


if not os.path.exists(path_rst):
    os.makedirs(path_rst)


print('%s'%(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())))



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('\ndevice-> ', device, '\n\n')


num_classes=600 # IITD: 460    KTU: 145    Tongji: 600    REST: 358    XJTU: 200  <---------

net = compnet(num_classes=num_classes)

# net.load_state_dict(torch.load('net_params.pkl'))

net.to(device)

#
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)  
scheduler = lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.8) 



# perform one epoch
def fit(epoch, model, data_loader, phase='training'):
    
    if phase != 'training' and phase != 'testing':
        raise TypeError('input error!')

    if phase == 'training':
        model.train()

    if phase == 'testing':
        model.eval()       
    

    running_loss = 0
    running_correct = 0

    for batch_id, (data, target) in enumerate(data_loader):  

        data = data.to(device)
        target = target.to(device)

        if phase == 'training':
            optimizer.zero_grad()
            output = model(data, target) 
        else:     
            with torch.no_grad():       
                output = model(data, None)   
     
        ce = criterion(output, target)    
        loss = ce
     

        ## log
        running_loss += loss.data.cpu().numpy() 

        preds = output.data.max(dim=1, keepdim=True)[1] # max returns (value, index)
        running_correct += preds.eq(target.data.view_as(preds)).cpu().sum().numpy()
        
        
        ## update
        if phase == 'training':
            loss.backward(retain_graph=None) #  
            optimizer.step()
           

    ## log info of this epoch
    total = len(data_loader.dataset)
    loss = running_loss / total
    accuracy = (100.0 * running_correct) / total

    if epoch % 10 == 0:
        print('epoch %d: \t%s loss is \t%7.5f ;\t%s accuracy is \t%d/%d \t%7.3f%%'%(epoch, phase, loss, phase, running_correct, total, accuracy))
        
    return loss, accuracy





train_losses, train_accuracy = [], []
val_losses, val_accuracy = [], []
bestacc = 0

for epoch in range(3000):

    epoch_loss, epoch_accuracy = fit(epoch, net, data_loader_train, phase='training')

    val_epoch_loss, val_epoch_accuracy = fit(epoch, net, data_loader_test, phase='testing') 

    scheduler.step()

    #------------------------logs----------------------
    train_losses.append(epoch_loss)
    train_accuracy.append(epoch_accuracy)

    val_losses.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)

    # save the best model
    if val_epoch_accuracy > bestacc:
        bestacc= val_epoch_accuracy
        torch.save(net.state_dict(), 'net_params_best.pth') 

    # save the current model and log info:
    if epoch % 10 == 0 or epoch == 2999 and epoch != 0:
        torch.save(net.state_dict(), 'net_params.pth') 

        plotLossACC(train_losses, val_losses, train_accuracy, val_accuracy)
        saveLossACC(train_losses, val_losses, train_accuracy, val_accuracy, bestacc)   
            

    # visualization
    if epoch % 200 == 0 or epoch == 2999: 
        saveFeatureMaps(net, data_loader_show, epoch)
        # printParameters(net)
        saveGaborFilters(net, epoch)
        saveParameters(net, epoch) 
        



# finished training
# torch.save(net.state_dict(), 'net_params.pth')
# torch.save(net, 'net.pkl') 

print('Finished Trainning')
print('the best testing acc is: ', bestacc, '%')
print('%s'%(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())))



# print('\n\n=======')
# print('testing ...')
# os.system('python test.py')


# print('%s'%(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())))