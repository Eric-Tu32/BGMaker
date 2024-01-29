import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import os

class MidiDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = os.listdir(data_dir)

    def __len__(self):
        return len(self.data_dir)

    def __getitem__(self, idx):
        arr = np.load(self.data_dir[idx])

        return arr

if __name__=="__main__":
    dir = "e:/Datasets/vgmidi-master/unlabelled/midi/midi"
    dataset = MidiDataset(dir)

    # Create a DataLoader
    batch_size = 2
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)