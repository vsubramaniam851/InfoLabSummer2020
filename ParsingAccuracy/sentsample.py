import csv
import random
import nltk
top_ten_speakers = ["Thor:", "Loki Actor:", "Valkyrie:", "Banner:", "Grandmaster:", "Hulk:", "Hela:", "Korg:", "Odin:", "Doctor Strange:"]
def extract_speakers(speakers):
	"""Takes in a list of speakers and extracts the associated sentences that they speak. 
	The associated sentences will be in a dictionary that maps the speaker to a list of all
	sentences they speak. """
	script = open("rag_script.txt", "r") #Open movie script
	script_lines = script.readlines() #Get all the lines from the movie 
	speaker_dict = {}
	for line in script_lines:
		lines_list = line.split() #Split each line into its words to get rid of blank space
		if lines_list:
			#Goal is to find word with ":" and get its index
			range1 = 0
			for i in range(len(lines_list)):
				if ":" in lines_list[i]: #Find word with ":" 
					speaker = lines_list[:i+1]
					range1 = i+1
			#Join the words to get a speaker
			speaker = ' '.join(speaker)
			if speaker in speakers:
				if speaker not in speaker_dict:
					#If speaker is in the dictionary, then add the sentence to the list for that speaker. Otherwise, create a new entry for that speaker.
					speaker_dict[speaker] = [' '.join(lines_list[range1:])]
				speaker_dict[speaker].append(' '.join(lines_list[range1:]))
	return speaker_dict

def random_sample_speaker(speakers, k):
	"""Get a sample size of k for each speaker and make a list of all sentences so that 
	we can feed sentence to the parser that we want to work with an visualize all the 
	sentences together."""
	out_dict = extract_speakers(speakers) #Get the speakers we are interested in
	sentences = {} #Contains the output sentences.
	for key in out_dict:
		sample = random.sample(out_dict[key], k) #Random sample k sentences for each speaker in the out_dict
		sentences[key] = sample 
	return sentences

def extract_lengths(filename):
	"""Takes in a movie script and extracts all sentences and maps each sentence to its corresponding sentence. It will return a dictionary that maps
	a length which is an int to a tuple containing the sentence and speaker. It uses NLTK sentence tokenization to break the paragraphs down into sentences
	since the script has only paragraphs."""
	length_dict = {} #Holds output
	script = open(filename, "r") #Open script
	script_lines = script.readlines()
	for line in script_lines:
		lines_list = line.split()
		if lines_list:
			range1 = 0 #Holds end of speaker
			for i in range(len(lines_list)):
				#find point where speaker label ends. Usually contains ":"
				if ":" in lines_list[i]:
					speaker = lines_list[:i+1]
					range1 = i+1
			speaker = ' '.join(speaker) 
			sent = ' '.join(lines_list[range1:]) #Outside of the speaker is the dialogue
			sent_list = nltk.sent_tokenize(sent) #Break dialogue into sentences
			for s in sent_list:
				s_list = s.split() #split into list to count number of words
				leng = len(s_list)
				try:
					length_dict[leng].append((s, speaker)) #If in dictionary, append
				except KeyError:
					length_dict[leng] = [(s, speaker)]
	return length_dict 

def random_sample_length(lengths, k, filename):
	"""Randomly samples based on length. Takes in a list of lengths to choose from, the number of sentences to sample, and a filename for the movie to sample from. It will check whether the
	sentenfce lengths exist and whether there are enough sentences with that length to sample k sentences and then take these sentences and add them to a sentence dict. Output will map
	the length to the sentence for the parser to parse. If the sentence length is not there, it will say sentence length is not present."""
	l_dict = extract_lengths(filename) #Get length dictionary
	sentences = {}
	for l in lengths:
		#For each length, check if the length exists in the dictionary using try/except
		try:
			out = l_dict[l]
			if len(out) > k: 
				#If there are more sentences than the number we want to sample, use random sample
				out_list = []
				sample = random.sample(out, k)
				for x in sample:
					out_list.append(x[0])
				sentences[l] = out_list
			else:
				#Otherwise, just take the entire list.
				out_list = []
				for x in out:
					out_list.append(x[0])
				sentences[l] = out_list
		except KeyError:
			print("This sentences length does not exist") #Throws exception and goes to next length. 
			continue
	return sentences

