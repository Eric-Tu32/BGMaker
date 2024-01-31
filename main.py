import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from datetime import datetime
import numpy as np
from matplotlib import pyplot as plt
import cv2
import gc

from dataloader import get_train_loader
from model import get_model
from train import train_model

gc.collect()
torch.cuda.empty_cache()

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = 0.2*BCE + 0.8*dice_loss
        
        return Dice_BCE

def save_model(model, optimizer, file_path):
    model.eval()
    optimizer.zero_grad()
    state = {
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict(),
    }

    torch.save(state, file_path)
    print('model saved')

print('Training Begins:')

train_dataloader = get_train_loader()

model = get_model()
model.to('cuda')

print("Num params: ", sum(p.numel() for p in model.parameters()))
for param in model.parameters():
    param.requires_grad = True
print("Weights Unfreezed, Proceeding: ")
print("Loss Function: DiceBCE + BCE")

#criterion = fl.losses.DiceLoss("binary")
loss_func = DiceBCELoss()
optimizer = torch.optim.Adam(model.parameters())

num_epochs = 100
print(f'Maximum Epoch = {num_epochs}')

train_model(model, train_dataloader, loss_func, optimizer, num_epochs = num_epochs, device = 'cuda')

#os.makedirs('./checkpoints', exist_ok=True)
#checkpoint_filename = f'E{num_epochs}_unet_112_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pth'
#save_model(model, optimizer, checkpoint_filename)
