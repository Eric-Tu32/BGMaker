import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import os

class MidiDataset(Dataset):
    def __init__(self, data_dir):
        self.dir_path = data_dir
        self.data_dir = os.listdir(data_dir)

    def __len__(self):
        return len(self.data_dir)

    def __getitem__(self, idx):
        path = os.path.join(self.dir_path, self.data_dir[idx])
        try:
            arr = np.load(path)
        except:
            print("sth's wrong, replacing with 0")
            arr = np.load(os.path.join(self.dir_path, self.data_dir[0]))
        return arr

def get_train_loader(batch_size = 32):
    dir = "/home/tbnrerk/mgen/processed_midi_data"
    ds = MidiDataset(dir)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)

if __name__=="__main__":
    dir = "/home/tbnrerk/mgen/processed_midi_data"
    dataset = MidiDataset(dir)
    
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    for data in dataset:
        print(data.shape, end="\r")
