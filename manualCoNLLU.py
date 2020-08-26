import os
import stanfordnlp
import nltk
import stanfordCoNLLU

def parse_list(base_path, movie_name, save_file = True):
	movie_name_manual = movie_name + "/manual"
	movie_dir_path = os.path.join(base_path, movie_name_manual)
	nlp = stanfordnlp.Pipeline()
	if os.path.exists(movie_dir_path):
		full_text = os.path.join(movie_dir_path, 'syntax-text.txt')
		f1 = open(full_text, "r")
		sents = nltk.sent_tokenize(f1.read())
		clean_sents = []
		for s in sents:
			# new_sent = s.replace("/", "")
			# new_sent = new_sent.replace("-", " ")
			# new_sent = new_sent.replace(":", " ")
			new_sent = s.replace("@", "")
			clean_sents.append(new_sent)
		# print(clean_sents)
		parses = stanfordCoNLLU.stanfordtodict(clean_sents)
		df_list = []
		for resp in parses:
			df_list.append(stanfordCoNLLU.dict_to_dataframe(resp))
		con_str = ''
		for df in df_list:
			out_string = stanfordCoNLLU.convert_dataframe_to_conllu(df)
			con_str = con_str + out_string
		if save_file:
			save_path = "/storage/vsub851/stanford-syntax"
			assert os.access(save_path, os.W_OK), 'Folder {} has no write permissions.'.format(movie_dir_path)
			save_name = movie_name + ".conllu"
			conllu_path = os.path.join(save_path, save_name)
			with open(conllu_path, 'w') as f:
				f.write(con_str)
		return con_str
	else:
		print("Movie path does not exist!")
		return None 

movie = ["ant-man", "aquaman", "avengers-infinity-war", "black-panther", "cars-2", "charlie-and-the-chocolate-factory", "coraline", "fantastic-mr-fox", "guardians-of-the-galaxy", "guardians-of-the-galaxy-2", "home-alone-2", "incredibles", "in-the-shadow-of-the-moon", "lotr-1", "lotr-2", "megamind", "sesame-street-episode-3990", "shrek-the-third", "spider-man-3-homecoming", "spider-man-far-from-home", "tank-chase", "the-martian", "toy-story", "toy-story-3-spanish", "venom", "wreck-it-ralph-spanish"]
movies_not_covered = []
for m in movie:
	out = parse_list("/storage/datasets/neuroscience/ecog/transcripts", m)
	if out == None:
		movies_not_covered.append(m)
print(movies_not_covered)
