import os
import numpy as np

dir_path = "/home/tbnrerk/mgen/processed_midi_data"
files = os.listdir(dir_path)

for f in files:
    path = os.path.join(dir_path, f)
    arr = np.load(path)
    print(np.max(arr))
    print(np.min(arr))
