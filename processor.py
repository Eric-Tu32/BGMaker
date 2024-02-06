import os
import numpy as np

data_dir = "/home/tbnrerk/mgen/midi_data"
data_files = os.listdir(data_dir)

original_length = len(data_files)
des_dir = "/home/tbnrerk/mgen/processed_midi_data"

counter = 0
for f in data_files:
    path = os.path.join(data_dir, f)
    arr = None
    try:
        arr = np.load(path)
    except:
        continue

    piece_length = 1920
    # Calculate the number of pieces
    num_pieces = arr.shape[0] // piece_length

    # Cut the array into pieces
    cut_pieces = [arr[i * piece_length : (i + 1) * piece_length] for i in range(num_pieces)]
    for i in cut_pieces:
        counter+=1
        print(f"{counter}/{original_length}", end="\r")
        des_path = os.path.join(des_dir, str(counter))
        np.save(des_path, i)
print("finish spliting")
