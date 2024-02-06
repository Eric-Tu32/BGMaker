import torch
import torch.nn as nn
import torch.nn.functional as F

import fusionlab as fl
from fusionlab.losses import DiceLoss, IoULoss

from matplotlib import pyplot as plt
from tqdm import tqdm
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

def train_model(model, train_loader, loss_func, optimizer, scheduler=None, num_epochs=10, device='cuda'):
    model.to(device)
    size = len(train_loader.dataset)
    epoch_train = {'loss': []}
    epoch_val = {'loss': []}
    best_performance = 0.0

    for epoch in range(num_epochs):
        gc.collect()
        torch.cuda.empty_cache()
        total_loss = 0.0

        # Use tqdm for a progress bar
        with tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False) as t:
            model.train()
            for batch in t:
                x = batch.float().to(device)
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(x)

                loss = loss_func(outputs, x)

                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Track metrics
                total_loss += loss.item()
                
                # Update the progress bar
                t.set_postfix(loss=loss.item())
                
        # Calculate average loss and accuracy for the epoch
        avg_loss = total_loss / len(train_loader)

        if (scheduler):
            scheduler.step()
              
        epoch_train['loss'].append(avg_loss)
        
        print(f'Epoch {epoch + 1}/{num_epochs}, Total Loss: {total_loss:.4f}, Avg Loss: {avg_loss:.4f}')
        
        if (1-avg_loss) > best_performance:
          print("New Best!")
          checkpoint_filename = "/home/tbnrerk/mgen/checkpoints/LSTMVAEV2.pth"
          save_model(model, optimizer, checkpoint_filename)
          best_performance = 1-avg_loss
        
      
    print('Training complete.')

# Plotting the training loss and accuracy
    plt.figure(figsize=(12, 5))

    # Plotting the training loss
    epochs = range(1, num_epochs + 1)
    plt.plot(epochs, epoch_train['loss'], 'b', label = 'Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.show()
