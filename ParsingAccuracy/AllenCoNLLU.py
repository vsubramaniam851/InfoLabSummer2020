import pandas as pd

def dict_to_dataframe(dict1):
	extracted_dict = {}
	extracted_dict["FORM"] = dict1["words"]
	extracted_dict["UPOS"] = dict1["pos"]
	extracted_dict["DEPREL"] = dict1["predicted_dependencies"]
	heads = []
	for i in dict1["predicted_heads"]:
		heads.append(str(i))
	extracted_dict["HEAD"] = heads
	underscore = []
	for i in range(0, len(dict1["words"])):
		underscore.append("_")
	extracted_dict["MISC"] = underscore
	extracted_dict["LEMMA"] = underscore
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
        conllu_str += '_\t'  # Mark empty for XPOS
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
