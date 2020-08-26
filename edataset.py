import h5py
import csv
import stanfordnlp
import numpy as np
def dataset(h5file, word_times_file, trigger_times_file):
	nlp = stanfordnlp.Pipeline()
	csv_file1 = open(word_times_file, "r")
	word_times = csv.reader(csv_file1, delimiter = ',')
	csv_file2 = open(trigger_times_file, "r")
	trigger_times = csv.reader(csv_file2, delimiter = ",")
	electrode_data = h5py.File(h5file, "r")['data']
	wordwithtime = []
	triggerwithindex = []
	for w in word_times:
		try:
			time = float(w[3]) - float(w[2])
		except ValueError:
			continue
		wordwithtime.append((w[1], time))
	for t in trigger_times:
		try:
			triggerwithindex.append((float(t[1]), int(t[4])))
		except ValueError:
			continue
	out_dict = interval(triggerwithindex, wordwithtime)
	index_dict = indices(out_dict)
	index_with_POS = {}
	dataset = []
	nouns = ["NNP", "NN", "NNS", "NNPS"]
	verbs = ["VB", "VBZ", "VBN", "VBP", "VBG", "VBD"]
	adjs = ["JJ", "JJR", "JJS"]
	for key in index_dict:
		doc = nlp(key)
		for sent in doc.sentences:
			for word in sent.words:
				if word.xpos in nouns:
					index_with_POS[(key, "N")] = index_dict[key]
				# elif word.xpos in verbs:
				# 	index_with_POS[(key, "V")] = index_dict[key]
				elif word.xpos in adjs:
					index_with_POS[(key, "A")] = index_dict[key]
	data_labels = {"N" : 0, "A": 1}
	for key in index_with_POS:
		index = index_with_POS[key]
		electrode_word = electrode_data[index]
		# electrode_word.resize((50, 50))
		electrode_word = np.resize(electrode_word, new_shape = 2500)
		# if key[1] == "V":
		# 	true_value = np.eye(3)[data_labels["V"]]
		if key[1] == "N":
			true_value = np.eye(2)[data_labels["N"]]
		elif key[1] == "A":
			true_value = np.eye(2)[data_labels["A"]]
		dataset.append([key[0], electrode_word, true_value])
	return dataset 

def interval(trigger_times, word_times):
	out_dict = {}
	for i in word_times:
		w = i[1]
		min_diff1 = float("inf")
		min_diff2 = float("inf")
		min_top = 0
		min_bott = 0
		for t in trigger_times:
			if t[0] > w:
				diff = t[0] - w
				if diff < min_diff1:
					min_diff1 = diff
					min_top = t
			if t[0] < w:
				diff = w - t[0]
				if diff < min_diff2:
					min_diff2 = diff
					min_bott = t
		out_dict[i] = [min_bott, min_top]
	return out_dict
def indices(in_dict):
	index_dict = {}
	for key in in_dict:
		value = in_dict[key]
		bottom = value[0]
		top = value[1]
		t_movie = key[1]
		t_pre = bottom[0]
		t_post = top[0]
		ind_pre = bottom[1]
		ind_post = top[1]
		index = int((t_movie-t_pre)/(t_post-t_pre) * (ind_post-ind_pre) + ind_pre)
		index_dict[key[0]] = index
	return index_dict


# h5file = "trial000/data.h5"
# word_times_file = "trial000/word-times.csv"
# trigger_times_file = "trial000/trigger-times.csv"
# data = dataset(h5file, word_times_file, trigger_times_file)
# # print(data)
# np.random.shuffle(data)
# # print(len(data[0][1]))
