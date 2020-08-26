import sentsample
from allennlp.predictors.predictor import Predictor
import allennlp_models.syntax.biaffine_dependency_parser
import nltk
import spaCyVis

predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/biaffine-dependency-parser-ptb-2020.04.06.tar.gz")
top_ten_speakers = ["Thor:", "Loki Actor:", "Valkyrie:", "Banner:", "Grandmaster:", "Hulk:", "Hela:", "Korg:", "Odin:", "Doctor Strange:"]
in_dict = sentsample.random_sample_speaker(top_ten_speakers, 5)
def predictions(dict1):
	"""Takes a dictionary of sentences with an associated speaker or length and parses each of them using the dependency parser. 
	Places them in a dictionary that has a key of the associated speaker or length."""
	out_dict = {} 
	for key in dict1: #Go through each speaker/length and each dialogue for that speaker/length
		for string in dict1[key]:
			#Break the dialogue into sentences since sometimes the dialogue can contain multiple sentences. This uses NLTK parsing.
			sent_list = nltk.sent_tokenize(string) 
			for sent in sent_list:
				prediction = predictor.predict(
  					sentence=sent
				) #Applying the parses from AllenNLP
				try:
					out_dict[key].append(prediction) #Place it in the output dictionary.
				except KeyError:
					out_dict[key] = [prediction]
	return out_dict
def AccuracyTest(type1, speakers, k, PDF = False):
	"""Main function to check accuracy. Input includes the sample type (length or speaker), a list of speakers if type1 is speakers, the sample size k, and 
	a PDF variable which states whether a PDF should be created or not for the visualization. The function then uses the prediction function to get the predictions and the 
	spaCyVis file to visualize the dependency tree for each sentence."""
	if type1 == "SPEAKERS": #Check type of sampling we want to do.
		in_dict = sentsample.random_sample_speaker(speakers, k)
		name = "BySpeaker.pdf" #Merged PDF
	all_predictions = predictions(in_dict) #get predictions
	html_dict = {}
	for key in all_predictions:
		#Take each prediction dictionary and get its corresponding HTML. 
		for pred in all_predictions[key]:
			html = spaCyVis.parser_to_input(pred)
			html_dict[html] = pred #Set the out dictionary to map HTML to the prediction dictionary for the out_pdf function
	html_list = html_dict.keys()
	if PDF:
		spaCyVis.out_pdf(html_dict, name) #If we want a MergedPDF as output, we call out_pdf from spaCyVis.
	return html_list #Return list of html which we can use to load the tree on a webpage.

AccuracyTest("SPEAKERS", top_ten_speakers, 5, PDF = True)


