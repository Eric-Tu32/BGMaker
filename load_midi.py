from midi_util import midi_to_np, np_to_midi
from mido import MidiFile

import numpy as np
import os

import logging
logging.basicConfig(filename='./loggers/load_midi.log', 
                    filemode='w', level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s : %(message)s'
                    )

dir_path = 'e:/Datasets/vgmidi-master/unlabelled/midi/midi'
data_dir = './midi_data'

for filename in os.listdir(dir_path):
    f = os.path.join(dir_path, filename)
    # checking if it is a file
    if os.path.isfile(f) and f.endswith(('mid','midi')):
        try:
            arr = midi_to_np(MidiFile(f))
        except:
            logging.warning(f'{f} is not saved')
        np.save(os.path.join(data_dir, filename.split(".")[0]), arr)
    