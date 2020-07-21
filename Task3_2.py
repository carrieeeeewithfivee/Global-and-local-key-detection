import numpy as np
from glob import glob
import pretty_midi
from librosa.feature import chroma_cqt
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score
from mir_eval.key import weighted_score
import scipy
import os
import utils

DB = 'A-MAPS'
Tonic_keys  = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
KS_MODE = {"major": [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    	   "minor": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]}

predictions, labels = list(), list()

for f in glob(DB+'/*.mid'):
	midi_data = pretty_midi.PrettyMIDI(f)
	song_lenth = int(midi_data.get_end_time())
	changes = midi_data.key_signature_changes
	key = 0
	last_second = 0
	for i in range(len(changes)):
		seconds = int(str(str(changes[i]).split(' ')[3]).split('.')[0])
		labels.extend([key]*(seconds-last_second))
		last_second = seconds
		key = str(str(changes[i]).split(' ')[0])+" "+str(str(changes[i]).split(' ')[1])
		key = int(utils.str_to_lerch(utils.generalize_key(key)))	
	labels.extend([key]*(song_lenth-seconds))

	chroms = np.asarray(midi_data.get_chroma())
	chroms = np.array_split(chroms,song_lenth,axis=1)

	for chrom in chroms:
		chroma_vector = np.sum(chrom, axis=1)
		try:
			max_r = 0
			for tonic in range(len(chroma_vector)):
				chroma_vector_rotate = utils.rotate(chroma_vector.tolist(), 12-tonic)
				correlation_major = scipy.stats.pearsonr(chroma_vector_rotate,KS_MODE['major'])[0]
				correlation_minor = scipy.stats.pearsonr(chroma_vector_rotate,KS_MODE['minor'])[0]
				if correlation_major>correlation_minor:
					if correlation_major > max_r:
						mode = "major"
						tonic_note = tonic
						max_r = correlation_major
				else:
					if correlation_minor > max_r:
						mode = "minor"
						tonic_note = tonic
						max_r = correlation_minor
			
			note = Tonic_keys[tonic_note] + " " + mode
			pred = utils.str_to_lerch(note)
			predictions.append(pred)
		except Exception as e:
			print(e)
			predictions.append(0)
	#break
score = 0
for i in range(len(labels)):
	score = score + weighted_score(utils.lerch_to_str(labels[i]), utils.lerch_to_str(predictions[i]))
weighted_accuracy = score/len(labels)
accuracy = accuracy_score(labels, predictions, normalize=True, sample_weight=None)

fout = open('output/Task3_2.txt','a')
fout.write("***** Task 3: local key detection *****\n")
fout.write("A-MAPS Overall Accuracy:\t{:.2f}%\n".format(accuracy*100))
fout.write("A-MAPS Overall Weighted Accuracy:\t{:.2f}%\n".format(weighted_accuracy*100))
fout.close()