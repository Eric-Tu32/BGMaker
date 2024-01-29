from mido import MidiFile, MidiTrack, Message
import numpy as np
from PIL import Image as im
import matplotlib.pyplot as plt
import os

def msg2dict(msg):
    result = dict()
    if msg.type == 'note_on':
        on_ = True
    elif msg.type ==  'note_off':
        on_ = False
    else:
        on_ = None
    result['time'] = int(msg.time)

    if on_ is not None:
        result['note'] = int(msg.note)
        result['velocity'] = int(msg.velocity)
    return [result, on_]
    
def trim_note(last_state, note, velocity, on_=True):
    '''
    return a list of 88 elements which represent one column in the timeline
    p.s. piano has 88 notes, corresponding to note id 21 to 108, any note out of this range will be ignored
    '''
    result = [0] * 88 if last_state is None else last_state.copy()
    if 21 <= note <= 108:
        result[note-21] = 1 if on_ else 0
    return result

def get_new_state(new_msg, last_state):
    new_msg, on_ = msg2dict(new_msg)
    new_state = trim_note(last_state, note=new_msg['note'], velocity=new_msg['velocity'], on_=on_) if on_ is not None else last_state
    return [new_state, new_msg['time']]

def track2seq(track, ticks_per_sample):
    # piano has 88 notes, corresponding to note id 21 to 108, any note out of the id range will be ignored
    abs_time = 0
    result = []
    last_state, _ = get_new_state(track[0], [0]*88)
    for i in range(1, len(track)):
        new_state, new_time = get_new_state(track[i], last_state)
        if new_time > 0:
            abs_time+=new_time
            if (new_time//ticks_per_sample>0):
                result += [last_state]*(new_time//ticks_per_sample)
            else:
                result += [last_state]
        last_state, _ = new_state, new_time
    return result

def mid2arry(mid, ticks_per_sample, min_msg_pct=0.1):
    tracks_len = [len(tr) for tr in mid.tracks]
    min_n_msg = max(tracks_len) * min_msg_pct
    # convert each track to nested list
    all_arys = []
    for i in range(len(mid.tracks)):
        if len(mid.tracks[i]) > min_n_msg:
            ary_i = track2seq(mid.tracks[i], ticks_per_sample)
            all_arys.append(ary_i)
    # make all nested list the same length
    max_len = max([len(ary) for ary in all_arys])
    for i in range(len(all_arys)):
        if len(all_arys[i]) < max_len:
            all_arys[i] += [[0] * 88] * (max_len - len(all_arys[i]))
    all_arys = np.array(all_arys, dtype='b')
    all_arys = all_arys.max(axis=0)
    # trim: remove consecutive 0s in the beginning and at the end
    sums = all_arys.sum(axis=1)
    ends = np.where(sums > 0)[0]
    return all_arys[min(ends): max(ends)]

def midi_to_np(midiFile):
    '''
    輸入一個Midi檔案，回傳一個np array
    '''
    samples_per_beat = 24
    ticks_per_beat = midiFile.ticks_per_beat #一拍幾個tick
    numerator = []    
    for track in midiFile.tracks:
        for msg in track:
            if (msg.type == 'time_signature'):
                numerator.append(int(msg.numerator))
                
    assert len(numerator) == 1

    ticks_per_sample = ticks_per_beat // samples_per_beat
    return mid2arry(midiFile, ticks_per_sample)

def np_to_midi(samples, ticks_per_beat=None, fname='MidiFile.mid', thresh=0.5):
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
 
def test():
    dir_path = 'e:/Datasets/vgmidi-master/unlabelled/midi/midi'
    file_name = 'Ace Attorney_Nintendo 3DS_Phoenix Wright Ace Attorney Spirit of Justice_Cheerful People'
    assert os.path.isfile(dir_path+'/'+file_name+'.mid')
    
    midiFile = MidiFile(dir_path+'/'+file_name+'.mid')
    result_array = midi_to_np(midiFile)
    np_to_midi(result_array)

if __name__ == '__main__':
    test()
