#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
error missing library: 
sudo apt-get update -y 
sudo apt-get install -y libsndfile1-dev

error cannot use sox
sudo apt-get update -y
sudo apt-get install -y sox
sudo apt-get install libsox-fmt-mp3
"""
from glob import glob
from collections import defaultdict
from librosa.feature import chroma_stft, chroma_cqt, chroma_cens
from scipy.stats import pearsonr
from mir_eval.key import weighted_score
from sklearn.metrics import accuracy_score
import scipy
import numpy as np
import utils

#templates
MODE    = {"major":[1,0,1,0,1,1,0,1,0,1,0,1],
           "minor":[1,0,1,1,0,1,0,1,1,0,1,0]}
KS_MODE = {"major": [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    	   "minor": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]}
KEY  = ['A','A#','B','C','C#','D','D#','E','F','F#','G','G#','a','a#','b','c','c#','d','d#','e','f','f#','g','g#']
Tonic_keys  = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

BPS = 'BPS_piano'
A_MAPS = 'A-MAPS'

def GTZAN(GTZAN_data,GTZAN_label,fout,Y = 0, weighted = False, KS = False):
	#parameters
	all_labels = defaultdict(list)
	all_predictions = defaultdict(list)
	accuracy = dict()

	#get GTZAN_data
	FILES = sorted(glob(GTZAN_data+'/*/*.wav'))
	LABELS = sorted(glob(GTZAN_label+'/*/*.lerch.txt'))
	GENRE = [g.split('/')[1] for g in glob(GTZAN_data+'/*')]
	FILELEN = len(FILES)
	assert len(FILES) == len(LABELS)

	#FILELEN = 10
	actual_len = 0
	for i in range(FILELEN):
		labels = int(utils.read_keyfile(LABELS[i],'*.lerch.txt')) #int
		if (int(labels)<0 or int(labels)>=24): continue
		genre_name = LABELS[i].split('/')[3]
		sr, y = utils.read_wav(FILES[i]) #sr:ex 22050 y:[ 0.00732422  0.01660156  0.00762939 ..
		all_labels[genre_name].append(labels)

		#create chromagram
		if Y == 0:
			chromagram = chroma_stft(y=y, sr=sr)
			#chromagram = chroma_cens(y=y, sr=sr)
			#chromagram = chroma_cqt(y=y, sr=sr)

		else:
			chromagram = np.log(1 + Y * np.abs(chroma_stft(y=y, sr=sr)))
			#chromagram = np.log(1 + Y * np.abs(chroma_cqt(y=y, sr=sr)))
			#chromagram = np.log(1 + Y * np.abs(chroma_cens(y=y, sr=sr)))
		chroma_vector = np.sum(chromagram, axis=1)
		#Template matching
		if KS==True:
			max_r = 0
			for tonic in range(len(chroma_vector)):
				chroma_vector_rotate = utils.rotate(chroma_vector.tolist(), 12-tonic)
				correlation_major = scipy.stats.pearsonr(chroma_vector_rotate,KS_MODE['major'])[0]
				correlation_minor = scipy.stats.pearsonr(chroma_vector_rotate,KS_MODE['minor'])[0]
				if correlation_major > correlation_minor:
					if correlation_major > max_r:
						mode = "major"
						tonic_note = tonic
						max_r = correlation_major
				else:
					if correlation_minor > max_r:
						mode = "minor"
						tonic_note = tonic
						max_r = correlation_minor
		else:
			#Find Tonic Note
			#tonic_note = np.argmax(chroma_vector)
			tonic_note = np.where(chroma_vector == np.amax(chroma_vector))
			tonic_note = int(tonic_note[0])
			chroma_vector = utils.rotate(chroma_vector.tolist(), 12 - tonic_note)
			correlation_major = scipy.stats.pearsonr(chroma_vector,MODE['major'])[0]
			correlation_minor = scipy.stats.pearsonr(chroma_vector,MODE['minor'])[0]
			mode = 'major' if correlation_major > correlation_minor else 'minor'

		#get prediction
		note = Tonic_keys[tonic_note] + " " + mode
		pred = utils.str_to_lerch(note)
		all_predictions[genre_name].append(pred)
		actual_len = actual_len + 1
	
	#calculate accuracy
	total_score = 0
	for gen in GENRE:
		if weighted == True:
			score = 0
			for i in range(len(all_labels[gen])):
				#print(all_labels[gen][i], all_predictions[gen][i])
				score = score + weighted_score(utils.lerch_to_str(all_labels[gen][i]), utils.lerch_to_str(all_predictions[gen][i]))
			if len(all_labels[gen])>0:
				accuracy[gen] = score/len(all_labels[gen])
			else:
				accuracy[gen] = 0
			total_score = total_score + score
		else:
			accuracy[gen] = accuracy_score(all_labels[gen], all_predictions[gen], normalize=True, sample_weight=None)
			total_score = total_score + accuracy_score(all_labels[gen], all_predictions[gen], normalize=False, sample_weight=None)

	total_accuracy = (total_score/actual_len)

	#write to file
	if KS == True: #Q2
		fout.write("***** Q2 Krumhansl-Schumuckler *****\n")
		if weighted==True:
			fout.write("***** Weighted *****\n")
		elif Y==0:
			fout.write("***** Normal *****\n")
		else:
			fout.write("***** Y: " +str(Y)+" *****\n")
		fout.write("GTZAN Genre    \taccuracy\n")
		for gen in GENRE:
			fout.write("{:9s}\t{:8.2f}%\n".format(gen,accuracy[gen]*100))
		fout.write("GTZAN Overall Accuracy:\t{:.2f}%\n".format(total_accuracy*100))

	else: #Q1
		if weighted==True:
			fout.write("***** Q1 Weighted *****\n")
		elif Y==0:
			fout.write("***** Q1 *****\n")
		else:
			fout.write("***** Q1 Y: " +str(Y)+" *****\n")
		fout.write("GTZAN Genre    \taccuracy\n")
		for gen in GENRE:
			fout.write("{:9s}\t{:8.2f}%\n".format(gen,accuracy[gen]*100))
		fout.write("GTZAN Overall Accuracy:\t{:.2f}%\n".format(total_accuracy*100))

def GIANTSTEPS(GIANTSTEPS_data,GIANTSTEPS_label,fout,Y = 0, weighted = False, KS = False):
	#parameters
	labels = []
	predictions = []
	#get Giantsteps_data
	FILES = sorted(glob(GIANTSTEPS_data+'/*.LOFI.wav'))
	LABELS = sorted(glob(GIANTSTEPS_label+'/key/*.LOFI.key'))
	FILELEN = len(FILES)
	assert len(FILES) == len(LABELS)
	#actual_len = 0
	#FILELEN = 5
	for i in range(FILELEN):
		label = utils.read_keyfile(LABELS[i],'*.LOFI.key')
		try:
			label = utils.generalize_key(label)
			label = int(utils.str_to_lerch(label))
		except:
			continue
		sr, y = utils.read_wav(FILES[i])
		labels.append(label)

		#create chromagram
		if Y == 0:
			chromagram = chroma_stft(y=y, sr=sr)
			#chromagram = chroma_cqt(y=y, sr=sr)
			#chromagram = chroma_cens(y=y, sr=sr)
		else:
			chromagram = np.log(1 + Y * np.abs(chroma_stft(y=y, sr=sr)))
			#chromagram = np.log(1 + Y * np.abs(chroma_cens(y=y, sr=sr)))
			#chromagram = np.log(1 + Y * np.abs(chroma_cqt(y=y, sr=sr)))
		chroma_vector = np.sum(chromagram, axis=1)
		#Template matching
		if KS==True:
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
		else:
			#Find Tonic Note
			#tonic_note = np.argmax(chroma_vector)
			tonic_note = np.where(chroma_vector == np.amax(chroma_vector))
			tonic_note = int(tonic_note[0])
			chroma_vector = utils.rotate(chroma_vector.tolist(), 12 - tonic_note)
			correlation_major = scipy.stats.pearsonr(chroma_vector,MODE['major'])[0]
			correlation_minor = scipy.stats.pearsonr(chroma_vector,MODE['minor'])[0]
			mode = 'major' if correlation_major > correlation_minor else 'minor'

		#get prediction
		note = Tonic_keys[tonic_note] + " " + mode
		pred = utils.str_to_lerch(note)
		predictions.append(pred)
		#actual_len = actual_len + 1

	#calculate accuracy
	score = 0
	if weighted == True:
		for i in range(len(labels)):
			score = score + weighted_score(utils.lerch_to_str(labels[i]), utils.lerch_to_str(predictions[i]))
		accuracy = score/len(labels)
	else:
		accuracy = accuracy_score(labels, predictions, normalize=True, sample_weight=None)


	#write to file
	if KS == True: #Q2
		fout.write("***** Q2 Krumhansl-Schumuckler *****\n")
		if weighted==True:
			fout.write("***** Weighted *****\n")
		elif Y==0:
			fout.write("***** Normal *****\n")
		else:
			fout.write("***** Y: " +str(Y)+" *****\n")
		fout.write("GIANTSTEPS Overall Accuracy:\t{:.2f}%\n".format(accuracy*100))
	else: #Q1
		if weighted==True:
			fout.write("***** Q1 Weighted *****\n")
		elif Y==0:
			fout.write("***** Q1 *****\n")
		else:
			fout.write("***** Q1 Y: " +str(Y)+" *****\n")
		fout.write("GIANTSTEPS Overall Accuracy:\t{:.2f}%\n".format(accuracy*100))


if __name__ == '__main__':

	GTZAN_data = 'GTZAN_wav'
	GTZAN_label = 'gtzan_key/gtzan_key/genres'
	GIANTSTEPS_data = 'giantsteps-key-dataset/audio'
	GIANTSTEPS_label = 'giantsteps-key-dataset/annotations'

	fout = open('output/Task1_Q1.txt','a')
	GTZAN(GTZAN_data,GTZAN_label,fout)
	GIANTSTEPS(GIANTSTEPS_data,GIANTSTEPS_label,fout)
	fout.close()

	#log (1 + γ|x|)
	fout = open('output/Task1_Q2.txt','a')
	GTZAN(GTZAN_data,GTZAN_label,fout,1)
	GTZAN(GTZAN_data,GTZAN_label,fout,10)
	GTZAN(GTZAN_data,GTZAN_label,fout,100)
	GTZAN(GTZAN_data,GTZAN_label,fout,1000)

	GIANTSTEPS(GIANTSTEPS_data,GIANTSTEPS_label,fout,1)
	GIANTSTEPS(GIANTSTEPS_data,GIANTSTEPS_label,fout,10)
	GIANTSTEPS(GIANTSTEPS_data,GIANTSTEPS_label,fout,100)
	GIANTSTEPS(GIANTSTEPS_data,GIANTSTEPS_label,fout,1000)
	fout.close()
	#weighted
	fout = open('output/Task1_Q3.txt','a')
	GTZAN(GTZAN_data,GTZAN_label,fout,weighted=True)
	GIANTSTEPS(GIANTSTEPS_data,GIANTSTEPS_label,fout,weighted=True)
	fout.close()
	#Task 2 Krumhansl-Schumuckler/Bonus
	fout = open('output/Task2_chroma_cens.txt','a')
	GTZAN(GTZAN_data,GTZAN_label,fout,KS=True)
	GIANTSTEPS(GIANTSTEPS_data,GIANTSTEPS_label,fout,KS=True)

	GTZAN(GTZAN_data,GTZAN_label,fout,Y=1,KS=True)
	GTZAN(GTZAN_data,GTZAN_label,fout,Y=10,KS=True)
	GTZAN(GTZAN_data,GTZAN_label,fout,Y=100,KS=True)
	GTZAN(GTZAN_data,GTZAN_label,fout,Y=1000,KS=True)

	GIANTSTEPS(GIANTSTEPS_data,GIANTSTEPS_label,fout,Y=1,KS=True)
	GIANTSTEPS(GIANTSTEPS_data,GIANTSTEPS_label,fout,Y=10,KS=True)
	GIANTSTEPS(GIANTSTEPS_data,GIANTSTEPS_label,fout,Y=100,KS=True)
	GIANTSTEPS(GIANTSTEPS_data,GIANTSTEPS_label,fout,Y=1000,KS=True)

	GTZAN(GTZAN_data,GTZAN_label,fout,weighted=True, KS=True)
	GIANTSTEPS(GIANTSTEPS_data,GIANTSTEPS_label,fout,weighted=True, KS=True)
	fout.close()



