from mido import MidiFile, MidiTrack, Message
import numpy as np
from PIL import Image as im

num_notes = 96 #使用96個音
samples_per_measure = 96 #一小節取96個sample

def midi_to_samples(fname):
    has_time_sig = False
    flag_warning = False
    mid = MidiFile(fname)
    ticks_per_beat = mid.ticks_per_beat #一拍幾個tick
    ticks_per_measure = 4 * ticks_per_beat #直接假設一小節四拍(為何)
    
    for i,track in enumerate(mid.tracks):
        for msg in track:
            if msg.type == 'time_signature':
                new_tpm = msg.numerator * ticks_per_beat * 4 / msg.denominator
                if has_time_sig and new_tpm != ticks_per_measure:
                    flag_warning = True
                ticks_per_measure = new_tpm
                has_time_sig = True
    if flag_warning:
        print('Warning: Detected multiple distinct time signatures')
        return []
        
    all_notes = {}
    for i, track in enumerate(mid.tracks):
        abs_time = 0 # 紀錄目前的時間，單位是tick
        for msg in track:
            abs_time += msg.time # msg.time是這個msg距離上個msg執行的時間間隔
            if msg.type == 'note_on':
                if msg.velocity == 0: # 等於note_off
                    continue
                note = msg.note - (128 - num_notes)/2 #midi有128個note，下縮到96個
                assert(note >= 0 and note < num_notes)
                if note not in all_notes: # 如果是新的note就放進all_note
                    all_notes[note] = []
                else: #如果已經存在就封閉上一個最新的note
                    if len(all_notes[note][-1]) == 1:
                        all_notes[note][-1].append(all_notes[note][-1][0] + 1)
                all_notes[note].append([(abs_time * samples_per_measure) // ticks_per_measure]) # append已過時間*96/1024*4，代表第幾個sample，可以把Sample當作新的時間單位
            elif msg.type == 'note_off':
                note = msg.note - (128 - num_notes)/2 #midi有128個note，下縮到96個
                assert(note >= 0 and note < num_notes)
                if len(all_notes[note][-1]) != 1:
                    continue
                all_notes[note][-1].append((abs_time * samples_per_measure) // ticks_per_measure)
        
    #處理只有開始的資料            
    for note in all_notes:
        for start_end in all_notes[note]:
            if len(start_end) == 1:
                start_end.append(start_end[0]+1)
                    
    samples = []
    for note in all_notes:
        for start, end in all_notes[note]:
            sample_ix = int(start // samples_per_measure)
            while len(samples) <= sample_ix:
                samples.append(np.zeros((samples_per_measure, num_notes), dtype=np.uint8))
            sample = samples[sample_ix]
            start_ix = int(start) - sample_ix * samples_per_measure
            
            sample[start_ix][int(note)] = 1
    print(len(samples))
    print(samples[0].sum())
    return samples
    
dir_path = 'e:/Datasets/vgmidi-master/unlabelled/midi/midi'
file_name = 'Ace Attorney_Nintendo 3DS_Phoenix Wright Ace Attorney Spirit of Justice_Cheerful People'  
samples = midi_to_samples(dir_path+'/'+file_name+'.mid')

full_midi = samples[0]
for i in range(1,len(samples)):
    full_midi = np.concatenate((full_midi, samples[i]), axis=1)

print(full_midi.shape)
    
data = im.fromarray(np.multiply(full_midi,255))
# saving the final output 
# as a PNG file
data.save('Ace Attorney_Nintendo 3DS_Phoenix Wright Ace Attorney Spirit of Justice_Cheerful People_full.png')