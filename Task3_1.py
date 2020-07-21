import numpy as np
from glob import glob
import pretty_midi
from librosa.feature import chroma_cqt
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score
from mir_eval.key import weighted_score
import scipy
import os

import utils # self-defined utils.py file
DB = 'BPS_piano'
Tonic_keys  = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
KS_MODE = {"major": [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    	   "minor": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]}

predictions, labels = list(), list()
for f in glob(DB+'/*.wav'):
	song_time = []
	key = utils.parse_key([line.split('\t')[1] for line in utils.read_BPS_keyfile(f,'REF_key_*.txt').split('\n')])
	try:
		key = list(map(lambda a: int(utils.str_to_lerch(utils.generalize_key(a))), key))
		labels.extend(key)
		song_time.extend(key)
	except:
		continue

	sr, y = utils.read_wav(f)
	chroms = np.asarray(chroma_cqt(y=y, sr=sr, hop_length=512))
	chroms = np.array_split(chroms,len(song_time),axis=1)

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
			predictions.append(0)

score = 0
for i in range(len(labels)):
	score = score + weighted_score(utils.lerch_to_str(labels[i]), utils.lerch_to_str(predictions[i]))
weighted_accuracy = score/len(labels)
accuracy = accuracy_score(labels, predictions, normalize=True, sample_weight=None)

fout = open('output/Task3_1.txt','a')
fout.write("***** Task 3: local key detection *****\n")
fout.write("BPS Overall Accuracy:\t{:.2f}%\n".format(accuracy*100))
fout.write("BPS Overall Weighted Accuracy:\t{:.2f}%\n".format(weighted_accuracy*100))
fout.close()