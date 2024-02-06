import torch
from model import get_model
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mido import MidiFile, MidiTrack, Message
import matplotlib.pyplot as plt

model = get_model()
checkpoint_path = "/home/tbnrerk/mgen/checkpoints/LSTMVAEV2.pth"
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model'])
model.eval()

decoder1 = model.decoder1
bn = model.bn3
decoder2 = model.decoder2
random_sample = torch.rand(1, 1920, 256)
h0 = torch.zeros(model.num_layers, 1, 128)
c0 = torch.zeros(model.num_layers, 1, 128)
z, _ = decoder2(random_sample, (h0,c0))
h0 = torch.zeros(model.num_layers, 1, 88)
c0 = torch.zeros(model.num_layers, 1, 88)
sheet, _ = decoder1(z, (h0,c0))

sheet = sheet.squeeze().detach().numpy()

# Define the threshold
threshold = 0.7

# Convert float array into a binary array based on the threshold
sheet = (sheet >= threshold).astype(int)

part_sheet = sheet[:96, :]
plt.imshow(part_sheet, cmap='gray')
plt.savefig(f"visualized_sheet.png")


def np_to_midi(samples, ticks_per_beat=None, fname='MidiFile.mid', thresh=0.2):
        if (fname.endswith(('.mid', '.midi'))):
                pass
        else:
                fname+='.mid'
        mid = MidiFile()
        track = MidiTrack()
        mid.tracks.append(track)
        if not ticks_per_beat:
                ticks_per_beat = mid.ticks_per_beat
        samples_per_beat = 24
        print(ticks_per_beat)
        ticks_per_sample = int(ticks_per_beat / samples_per_beat)
        abs_time = 0
        last_time = 0
        for col in range(samples.shape[0]):
                abs_time += ticks_per_sample
                for note_idx in range(samples.shape[1]):
                        if samples[col][note_idx] >= thresh and (col == 0 or samples[col-1,note_idx] < thresh):
                                delta_time = abs_time - last_time
                                track.append(Message('note_on', note=int(note_idx+21), velocity=127, time=delta_time))
                                last_time = abs_time
                        if samples[col][note_idx] >= thresh and (col == samples.shape[0]-1 or samples[col+1][note_idx] < thresh):
                                delta_time = abs_time - last_time
                                track.append(Message('note_off', note=int(note_idx+21), velocity=127, time=delta_time))
                                last_time = abs_time
        mid.save(fname)
np_to_midi(sheet)

