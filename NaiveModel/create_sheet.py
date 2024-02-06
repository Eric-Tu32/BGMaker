import torch
from model import get_model
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mido import MidiFile, MidiTrack, Message

model = get_model()
checkpoint_path = "/home/tbnrerk/mgen/checkpoints/Naive2.pth"
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model'])
model.eval()

class InterpolateModule(nn.Module):
    def __init__(self, scale_factor):
        super(InterpolateModule, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)

decoder = nn.Sequential(
    model.decoder4,
    InterpolateModule(scale_factor=2),
    model.decoder3,
    InterpolateModule(scale_factor=2),
    model.decoder2,
    InterpolateModule(scale_factor=2),
    model.output,
)
random_sample = torch.rand(1,256, 240, 11)

sheet = decoder(random_sample).squeeze().detach().numpy()
threshold = 0.5
random_indices = np.random.rand(sheet.shape[0], sheet.shape[1]) < threshold
sheet[random_indices] = 1
print(sheet.shape)

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
