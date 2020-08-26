import stanfordnlp
import pandas as pd
import sentsample

def stanfordtodict(sents):
	nlp = stanfordnlp.Pipeline()
	dicts = []
	for s in sents:
		doc = nlp(s)
		dep_list = []
		for sent in doc.sentences:
			for dep in sent.dependencies:
				# print((dep[2].text, dep[0].index, dep[1]))
				dep_list.append((dep[2].text, dep[0].index, dep[1]))
				# print(dep_list)
		out_dict = {}
		for d in dep_list:
			try:
				out_dict["words"].append(d[0])
			except KeyError:
				out_dict["words"] = [d[0]]				
			try:
				out_dict["predicted_dependencies"].append(d[2])
			except KeyError:
					out_dict["predicted_dependencies"] = [d[2]]
			try:
				out_dict["predicted_heads"].append(d[1])
			except KeyError:
				out_dict["predicted_heads"] = [d[1]]
		for sent in doc.sentences:
			for word in sent.words:
				try:
					out_dict["XPOS"].append(word.xpos)
				except KeyError:
					out_dict["XPOS"] = [word.xpos]
				try:
					out_dict["UPOS"].append(word.upos)
				except KeyError:
					out_dict["UPOS"] = [word.upos]
				try:
					out_dict["lemma"].append(word.lemma)
				except:
					out_dict["lemma"] = [word.lemma]
		dicts.append(out_dict)
	return dicts
# sents = ["This is the best parser ever!", "There are no mistakes at all!", "Nothing to be seen here!", "Wait... how could there not be a mistake?"]
# dicts = stanfordtodict(sents)
# print(dicts)
def dict_to_dataframe(dict1):
	extracted_dict = {}
	extracted_dict["FORM"] = dict1["words"]
	# print("WORDS", len(extracted_dict["FORM"]))
	extracted_dict["DEPREL"] = dict1["predicted_dependencies"]
	# print(len(extracted_dict["DEPREL"]))
	extracted_dict["HEAD"] = dict1["predicted_heads"]
	# print(len(extracted_dict["HEAD"]))
	underscore = []
	for i in range(0, len(dict1["words"])):
		underscore.append("_")
	extracted_dict["MISC"] = underscore
	extracted_dict["LEMMA"] = underscore
	extracted_dict["UPOS"] = dict1["UPOS"]
	# print("POS", len(extracted_dict["UPOS"]))
	extracted_dict["XPOS"] = dict1["XPOS"]
	extracted_dict["LEMMA"] = dict1["lemma"]
	resp_df = pd.DataFrame.from_dict(extracted_dict)
	return resp_df

def convert_dataframe_to_conllu(df, conllu_str=''):
    line_num = 1
    prev_end = 0 
    for i in df.index:
        '''
        if int(df.loc[i, 'HEAD'] + 1 - prev_end) < 0: 
            prev_end += line_num
            line_num = 1 
            continue
        ''' 
        row = df.loc[i]
        conllu_str += str(line_num) + '\t'
        conllu_str += df.loc[i, 'FORM'] + '\t'
        conllu_str += df.loc[i, 'LEMMA'] + '\t'
        conllu_str += df.loc[i, 'UPOS'] + '\t'
        conllu_str += df.loc[i, 'XPOS'] + '\t'
        conllu_str += '_\t'  # Mark empty for FEATS
        conllu_str += df.loc[i, 'HEAD'] + '\t'
        # if not df.loc[i, 'DEPREL'] == 'root':   
        #     conllu_str += str(int(df.loc[i, 'HEAD'] + 1 - prev_end)) + '\t'
        # else:
        #     conllu_str += str(0) + '\t'
        conllu_str += df.loc[i, 'DEPREL'] + '\t'
        conllu_str += '_\t'  # Mark empty for DEPS
        conllu_str += df.loc[i, 'MISC']
        #conllu_str += '_'  # Mark empty for MISC
        conllu_str += '\n'
        line_num += 1
        
        # if df.loc[i, 'FORM'] in ['?', '!', '.', '...', '. . .', '…', '....']:
        #     conllu_str += '\n'
        #     prev_end += line_num
        #     line_num = 1            
        # else:
        #     line_num += 1
    conllu_str += '\n'
            
    return conllu_str

def parse_list(list1):
	df_list = []
	for resp in list1:
		df_list.append(dict_to_dataframe(resp))
	con_str = ''
	for df in df_list:
		out_string = convert_dataframe_to_conllu(df)
		con_str = con_str + out_string
	return con_str
# print(parse_list(dicts))


# top_ten_speakers = ["Thor:", "Loki Actor:", "Valkyrie:", "Banner:", "Grandmaster:", "Hulk:", "Hela:", "Korg:", "Odin:", "Doctor Strange:"]
# lengths = [3, 23, 35, 2, 4, 10]
# list1 = []
# in_dict = sentsample.random_sample_speaker(top_ten_speakers, 2)
# in_dict2 = sentsample.random_sample_length(lengths, 5, "rag_script.txt")
# for key in in_dict2:
# 	list1 = list1 + stanfordtodict(in_dict2[key])
# print(parse_list(list1))



