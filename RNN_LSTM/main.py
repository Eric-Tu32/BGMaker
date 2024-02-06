import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
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
        #inputs = F.sigmoid(inputs)       

        #flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = 0.5*BCE + 0.5*dice_loss
        
        return Dice_BCE

class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()

    def forward(self, inputs, targets):
        inputs = F.sigmoid(inputs)
        return F.binary_cross_entropy(inputs, targets)


class L2BCELoss(nn.Module):
    def __init__(self):
        super(L2BCELoss, self).__init__()

    def forward(self, inputs, targets):
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)

        bce = F.binary_cross_entropy(inputs, targets, reduction='mean')
        l2 = F.mse_loss(inputs, targets)

        return 0.1*bce + l2

print('Training Begins:')

train_dataloader = get_train_loader()

model = get_model()
model.to('cuda')

print("Num params: ", sum(p.numel() for p in model.parameters()))
for param in model.parameters():
    param.requires_grad = True

#criterion = fl.losses.DiceLoss("binary")
loss_func = nn.MSELoss()
#loss_func = DiceBCELoss()
#loss_func = BCELoss()
#loss_func = L2BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

load_model = False
if load_model:
    checkpoint_path = "/home/tbnrerk/mgen/checkpoints/RNNVAE.pth"
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

num_epochs = 100
print(f'Maximum Epoch = {num_epochs}')

train_model(model, train_dataloader, loss_func, optimizer, num_epochs = num_epochs, device = 'cuda')

#os.makedirs('./checkpoints', exist_ok=True)
#checkpoint_filename = f'E{num_epochs}_unet_112_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pth'
#save_model(model, optimizer, checkpoint_filename)
