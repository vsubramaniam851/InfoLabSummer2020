import sentsample
from allennlp.predictors.predictor import Predictor
import allennlp_models.syntax.biaffine_dependency_parser
import nltk
import CoNLLU

top_ten_speakers = ["Thor:", "Loki Actor:", "Valkyrie:", "Banner:", "Grandmaster:", "Hulk:", "Hela:", "Korg:", "Odin:", "Doctor Strange:"] #Top ten most common speakers
predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/biaffine-dependency-parser-ptb-2020.04.06.tar.gz")
def predictions(dict1):
	"""Takes a dictionary of sentences with an associated speaker or length and parses each of them using the dependency parser. 
	Places them in a dictionary that has a key of the associated speaker or length."""
	out_list = [] 
	for key in dict1: #Go through each speaker/length and each dialogue for that speaker/length
		for string in dict1[key]:
			#Break the dialogue into sentences since sometimes the dialogue can contain multiple sentences. This uses NLTK parsing.
			sent_list = nltk.sent_tokenize(string) 
			for sent in sent_list:
				prediction = predictor.predict(
  					sentence=sent
				) #Applying the parses from AllenNLP
				out_list.append(prediction)
	return out_list
def AccuracyTest(type1, k, filename, speakers = None, lengths = None, f_name = None):
	"""Main function to check accuracy. Input includes the sample type (length or speaker), a list of speakers if type1 is speakers and the sample size k. The output is a CoNLLU string
	as well as a text file which contains a CoNLL-U string which can be visualized using the CoNLL-U editor package. Uses the CoNLL-U script imported to do so. Writes
	CoNLLU to filename and uses the script from f_name which may or may not be needed depending on function."""
	if type1 == "SPEAKERS": #Check type of sampling we want to do.
		in_dict = sentsample.random_sample_speaker(speakers, k)
	elif type1 == "LENGTHS":
		in_dict = sentsample.random_sample_length(lengths, k, f_name)
	all_predictions = predictions(in_dict) #get predictions
	out_string = CoNLLU.parse_list(all_predictions) #Get ConLLU for these predictions
	with open(filename, "w") as f:
		f.write(out_string) #Write to out file
	return out_string

# AccuracyTest("SPEAKERS", 5, "SpeakerAccuracy.conllu", top_ten_speakers)

AccuracyTest(type1 = "LENGTHS", k = 3, filename = "LenAccuracy.conllu", lengths = [3, 23, 35, 2, 4, 10], f_name = "rag_script.txt")
