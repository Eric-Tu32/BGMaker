import torch
import torch.nn as nn
import torch.nn.functional as F

import fusionlab as fl
from fusionlab.losses import DiceLoss, IoULoss

from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
from datetime import datetime
import gc

from dataloader import get_train_loader

def save_model(model, optimizer, file_path):
    model.eval()
    optimizer.zero_grad()
    state = {
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict(),
    }

    torch.save(state, file_path)
    print(f'checkpoint model saved to {file_path}')

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
        
def get_RP(predicted, targets):
    TP, FP, TN, FN = 0, 0, 0, 0
    for i in range(len(predicted)):
        if (predicted[i][0]==1.0 and targets[i][0]==1.0): TP+=1 
        if (predicted[i][0]==1.0 and targets[i][0]==0.0): FP+=1 
        if (predicted[i][0]==0.0 and targets[i][0]==1.0): FN+=1 
        if (predicted[i][0]==0.0 and targets[i][0]==0.0): TN+=1 
    return TP, FP, TN, FN

def get_F1_score(TP, FP, TN, FN):
    recall = TP / (TP + FN) if TP+FN != 0 else 0 
    precision = TP / (TP + FP) if TP+FP !=0 else 0
    F1_score = 2 * (recall * precision) / (recall + precision) if recall + precision != 0 else 0 
    return F1_score
    

def validate(model, val_loader, seg_criterion, class_criterion, device = 'cuda'):
  size = len(val_loader.dataset)
  num_batches = len(val_loader)
  model.eval()
  epoch_loss, epoch_dice = 0, 0
  epoch_correct = 0
  ttp, tfp, ttn, tfn = 0, 0, 0, 0

  with torch.no_grad():
    for batch_i, batch in enumerate(val_loader):
      data_names, inputs, masks, labels = batch['data_name'], batch['data'].to(device), batch['mask'].to(device), batch['label'].to(device)
      inputs = inputs.float()
      outputs, predicted_logits = model(inputs)
      
      loss = seg_criterion(outputs, masks)
      loss += 0.4*class_criterion(predicted_logits, labels)
      epoch_loss += loss.item()
      
      
      probabilities = F.sigmoid(predicted_logits)
      predicted_labels = torch.where(probabilities > 0.5, torch.tensor(1.0, dtype=torch.float64), torch.tensor(0.0, dtype=torch.float64))
      epoch_correct += predicted_labels.eq(labels).sum().item()
      tp, fp, tn, fn = get_RP(predicted_labels, labels)
      ttp += tp
      tfp += fp
      ttn += tn
      tfn += fn
    
    avg_loss = epoch_loss / num_batches
    f1 = get_F1_score(ttp, tfp, ttn, tfn)
    recall = ttp/(ttp+tfn+0.00001)
    prec = ttp/(ttp+tfp+0.00001)
    accuracy = epoch_correct / size
    
    return avg_loss, accuracy, f1, recall, prec

def train_model(model, train_loader, loss_func, optimizer, num_epochs=10, device='cuda'):
    model.to(device)
    size = len(train_loader.dataset)
    epoch_train = {'loss': []}
    epoch_val = {'loss': []}
    best_performance = 0
    
    for epoch in range(num_epochs):
        gc.collect()
        torch.cuda.empty_cache()
        total_loss = 0.0
        total_dice = 0.0
        epoch_correct = 0.0

        # Use tqdm for a progress bar
        with tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False) as t:
            model.train()
            for batch in t:
                x = batch.to(device)
                optimizer.zero_grad()
                
                # Forward pass
                inputs = x.float().unsqueeze(1)
                outputs = model(inputs)
                loss = loss_func(outputs, inputs)

                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Track metrics
                total_loss += loss.item()
                
                probabilities = F.sigmoid(outputs)
                predicted = torch.where(probabilities > 0.5, torch.tensor(1.0, dtype=torch.float64), torch.tensor(0.0, dtype=torch.float64))

                # Update the progress bar
                t.set_postfix(loss=loss.item())
                
        # Calculate average loss and accuracy for the epoch
        avg_loss = total_loss / len(train_loader)
              
        epoch_train['loss'].append(avg_loss)
        
        print(f'Epoch {epoch + 1}/{num_epochs}, Avg Loss: {avg_loss:.4f}')
        
        if (1-avg_loss) > best_performance:
          print("New Best!")
          checkpoint_filename = "/home/tbnrerk/mgen/checkpoints/Naive.pth"
          save_model(model, optimizer, checkpoint_filename)
          best_performance = 1-avg_loss
        
      
    print('Training complete.')

# Plotting the training loss and accuracy
    plt.figure(figsize=(12, 5))

    # Plotting the training loss
    epochs = range(1, num_epochs + 1)
    plt.plot(epochs, epoch_train['loss'], 'b', label = 'Training Loss')
    plt.plot(epochs, epoch_val['loss'], 'r', label = 'Validation Loss')
    plt.title('Training Loss and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.show()
