from tkinter import *
from tkinter import messagebox
from tkinter import ttk 
def visualizer(dict1):
    root = Tk()
    m1 = PanedWindow(root)
    paned_window = PanedWindow(m1, orient = VERTICAL, showhandle = True, sashrelief = RAISED)
    m1.pack(fill=BOTH, expand=1)
    sentence = ' '.join(dict1["words"])
    top = Label(paned_window, text = sentence, font = ('arial', 15, 'bold'), bg = 'orange')
    
    paned_window.add(top)
    def showDeps():
        words = dict1["words"]
        predicted_heads = dict1["predicted_heads"]
        heads = {}
        for i in range(0, len(predicted_heads)):
            num = predicted_heads[i]
            if num != 0:
                word = words[num-1]
                try:
                    heads[word].append(words[i])
                except KeyError:
                    heads[word] = [words[i]]
        out_string = "DEPENDENCIES:\n"
        for key in heads:
            out_string = out_string + key + ": " + '/'.join(heads[key]) + "\n"
        bottom = Label(paned_window, text = out_string, font = ('arial', 15), bg = "pink")
        paned_window.add(bottom)
    def showHeads():
        words = dict1["words"]
        predicted_heads = dict1["predicted_heads"]
        heads = []
        for num in predicted_heads:
            if num != 0:
                if words[num-1] not in heads:
                    heads.append(words[num - 1])
        out_string = "HEADS:\n"
        for head in heads:
            out_string = out_string + head + "\n"
        bottom = Label(paned_window, text = out_string, font = ('arial', 15), bg = "red")
        paned_window.add(bottom)
    def showTails():
        words = dict1["words"]
        predicted_heads = dict1["predicted_heads"]
        heads = {}
        for i in range(0, len(predicted_heads)):
            num = predicted_heads[i]
            if num != 0:
                word = words[num-1]
                try:
                    heads[word].append(words[i])
                except KeyError:
                    heads[word] = [words[i]]
        out_string = "TAILS:\n"
        for head in heads:
            for tail in heads[head]:
                out_string = out_string + tail + ": " + head + "\n"
        bottom = Label(paned_window, text = out_string, font = ('arial', 15), bg = "cyan")
        paned_window.add(bottom)
    m1.add(paned_window)
    menu_bar = Menu(root)
    menu_bar.add_command(label = "Deps", command = showDeps)
    menu_bar.add_command(label = "Heads", command = showHeads)
    menu_bar.add_command(label = "Tails", command = showTails)
    root.config(menu = menu_bar)
    root.mainloop()
    
    
            


# predicted_heads': [3, 3, 10, 5, 3, 3, 10, 10, 10, 0, 10, 11, 10]
# ['If', 'I', 'bring', '10', 'dollars', 'tomorrow', ',', 'can', 'you', 'buy', 'me', 'lunch', '?']

# in_dict = {'arc_loss': 0.853976845741272, 'tag_loss': 0.4015558958053589, 'loss': 1.2555327415466309, 
# 'words': ['Here', 'is', 'a', 'text', 'file', '.', 'It', 'will', 'take', 'these', 'sentences', 'and', 'print', 'them', 'out', 'into', 'several', 'lines', '.', 'Hopefully', 'it', 'works', 'and', 'does', 'not', 'cause', 'any', 'problems', '.', 'It', 'should', 'also', 'skip', 'the', 'blank', 'lines', 'hopefully', '?', 'I', 'hope', 'so', '!', '!', 'I', 'think', 'so', '!', 'Please', 'work', '!'], 
# 'pos': ['ADV', 'VERB', 'DET', 'NOUN', 'NOUN', 'PUNCT', 'PRON', 'VERB', 'VERB', 'DET', 'NOUN', 'CCONJ', 'VERB', 'PRON', 'PART', 'ADP', 'ADJ', 'NOUN', 'PUNCT', 'ADV', 'PRON', 'VERB', 'CCONJ', 'VERB', 'ADV', 'ADP', 'DET', 'NOUN', 'PUNCT', 'PRON', 'VERB', 'ADV', 'VERB', 'DET', 'ADJ', 'NOUN', 'ADV', 'PUNCT', 'PRON', 'VERB', 'ADV', 'PUNCT', 'PUNCT', 'PRON', 'VERB', 'ADV', 'PUNCT', 'INTJ', 'VERB', 'PUNCT'], 
# 'predicted_dependencies': ['advmod', 'root', 'det', 'dep', 'dep', 'dep', 'nsubj', 'aux', 'dep', 'dep', 'dobj', 'dep', 'dep', 'dep', 'dep', 'prep', 'amod', 'dep', 'dep', 'dep', 'dep', 'dep', 'cc', 'aux', 'neg', 'dep', 'amod', 'dobj', 'dep', 'nsubj', 'aux', 'dep', 'dep', 'det', 'amod', 'dep', 'dep', 'dep', 'dep', 'dep', 'dep', 'dep', 'dep', 'dep', 'dep', 'dep', 'dep', 'dep', 'dep', 'punct'], 
# 'predicted_heads': [2, 0, 5, 5, 2, 5, 9, 9, 2, 11, 9, 13, 11, 13, 14, 15, 18, 16, 18, 22, 22, 18, 26, 26, 26, 9, 28, 33, 33, 33, 33, 33, 26, 36, 36, 33, 33, 40, 40, 33, 40, 40, 45, 45, 40, 45, 48, 45, 48, 48], 
# 'hierplane_tree': {'text': 'Here is a text file . It will take these sentences and print them out into several lines . Hopefully it works and does not cause any problems . It should also skip the blank lines hopefully ? I hope so ! ! I think so ! Please work !', 'root': {'word': 'is', 'nodeType': 'root', 'attributes': ['VERB'], 'link': 'root', 'spans': [{'start': 5, 'end': 8}], 'children': [{'word': 'Here', 'nodeType': 'advmod', 'attributes': ['ADV'], 'link': 'advmod', 'spans': [{'start': 0, 'end': 5}]}, {'word': 'file', 'nodeType': 'dep', 'attributes': ['NOUN'], 'link': 'dep', 'spans': [{'start': 15, 'end': 20}], 'children': [{'word': 'a', 'nodeType': 'det', 'attributes': ['DET'], 'link': 'det', 'spans': [{'start': 8, 'end': 10}]}, {'word': 'text', 'nodeType': 'dep', 'attributes': ['NOUN'], 'link': 'dep', 'spans': [{'start': 10, 'end': 15}]}, {'word': '.', 'nodeType': 'dep', 'attributes': ['PUNCT'], 'link': 'dep', 'spans': [{'start': 20, 'end': 22}]}]}, {'word': 'take', 'nodeType': 'dep', 'attributes': ['VERB'], 'link': 'dep', 'spans': [{'start': 30, 'end': 35}], 'children': [{'word': 'It', 'nodeType': 'nsubj', 'attributes': ['PRON'], 'link': 'nsubj', 'spans': [{'start': 22, 'end': 25}]}, {'word': 'will', 'nodeType': 'aux', 'attributes': ['VERB'], 'link': 'aux', 'spans': [{'start': 25, 'end': 30}]}, {'word': 'sentences', 'nodeType': 'dobj', 'attributes': ['NOUN'], 'link': 'dobj', 'spans': [{'start': 41, 'end': 51}], 'children': [{'word': 'these', 'nodeType': 'dep', 'attributes': ['DET'], 'link': 'dep', 'spans': [{'start': 35, 'end': 41}]}, {'word': 'print', 'nodeType': 'dep', 'attributes': ['VERB'], 'link': 'dep', 'spans': [{'start': 55, 'end': 61}], 'children': [{'word': 'and', 'nodeType': 'dep', 'attributes': ['CCONJ'], 'link': 'dep', 'spans': [{'start': 51, 'end': 55}]}, {'word': 'them', 'nodeType': 'dep', 'attributes': ['PRON'], 'link': 'dep', 'spans': [{'start': 61, 'end': 66}], 'children': [{'word': 'out', 'nodeType': 'dep', 'attributes': ['PART'], 'link': 'dep', 'spans': [{'start': 66, 'end': 70}], 'children': [{'word': 'into', 'nodeType': 'prep', 'attributes': ['ADP'], 'link': 'prep', 'spans': [{'start': 70, 'end': 75}], 'children': [{'word': 'lines', 'nodeType': 'dep', 'attributes': ['NOUN'], 'link': 'dep', 'spans': [{'start': 83, 'end': 89}], 'children': [{'word': 'several', 'nodeType': 'amod', 'attributes': ['ADJ'], 'link': 'amod', 'spans': [{'start': 75, 'end': 83}]}, {'word': '.', 'nodeType': 'dep', 'attributes': ['PUNCT'], 'link': 'dep', 'spans': [{'start': 89, 'end': 91}]}, {'word': 'works', 'nodeType': 'dep', 'attributes': ['VERB'], 'link': 'dep', 'spans': [{'start': 104, 'end': 110}], 'children': [{'word': 'Hopefully', 'nodeType': 'dep', 'attributes': ['ADV'], 'link': 'dep', 'spans': [{'start': 91, 'end': 101}]}, {'word': 'it', 'nodeType': 'dep', 'attributes': ['PRON'], 'link': 'dep', 'spans': [{'start': 101, 'end': 104}]}]}]}]}]}]}]}]}, {'word': 'cause', 'nodeType': 'dep', 'attributes': ['ADP'], 'link': 'dep', 'spans': [{'start': 123, 'end': 129}], 'children': [{'word': 'and', 'nodeType': 'cc', 'attributes': ['CCONJ'], 'link': 'cc', 'spans': [{'start': 110, 'end': 114}]}, {'word': 'does', 'nodeType': 'aux', 'attributes': ['VERB'], 'link': 'aux', 'spans': [{'start': 114, 'end': 119}]}, {'word': 'not', 'nodeType': 'neg', 'attributes': ['ADV'], 'link': 'neg', 'spans': [{'start': 119, 'end': 123}]}, {'word': 'skip', 'nodeType': 'dep', 'attributes': ['VERB'], 'link': 'dep', 'spans': [{'start': 159, 'end': 164}], 'children': [{'word': 'problems', 'nodeType': 'dobj', 'attributes': ['NOUN'], 'link': 'dobj', 'spans': [{'start': 133, 'end': 142}], 'children': [{'word': 'any', 'nodeType': 'amod', 'attributes': ['DET'], 'link': 'amod', 'spans': [{'start': 129, 'end': 133}]}]}, {'word': '.', 'nodeType': 'dep', 'attributes': ['PUNCT'], 'link': 'dep', 'spans': [{'start': 142, 'end': 144}]}, {'word': 'It', 'nodeType': 'nsubj', 'attributes': ['PRON'], 'link': 'nsubj', 'spans': [{'start': 144, 'end': 147}]}, {'word': 'should', 'nodeType': 'aux', 'attributes': ['VERB'], 'link': 'aux', 'spans': [{'start': 147, 'end': 154}]}, {'word': 'also', 'nodeType': 'dep', 'attributes': ['ADV'], 'link': 'dep', 'spans': [{'start': 154, 'end': 159}]}, {'word': 'lines', 'nodeType': 'dep', 'attributes': ['NOUN'], 'link': 'dep', 'spans': [{'start': 174, 'end': 180}], 'children': [{'word': 'the', 'nodeType': 'det', 'attributes': ['DET'], 'link': 'det', 'spans': [{'start': 164, 'end': 168}]}, {'word': 'blank', 'nodeType': 'amod', 'attributes': ['ADJ'], 'link': 'amod', 'spans': [{'start': 168, 'end': 174}]}]}, {'word': 'hopefully', 'nodeType': 'dep', 'attributes': ['ADV'], 'link': 'dep', 'spans': [{'start': 180, 'end': 190}]}, {'word': 'hope', 'nodeType': 'dep', 'attributes': ['VERB'], 'link': 'dep', 'spans': [{'start': 194, 'end': 199}], 'children': [{'word': '?', 'nodeType': 'dep', 'attributes': ['PUNCT'], 'link': 'dep', 'spans': [{'start': 190, 'end': 192}]}, {'word': 'I', 'nodeType': 'dep', 'attributes': ['PRON'], 'link': 'dep', 'spans': [{'start': 192, 'end': 194}]}, {'word': 'so', 'nodeType': 'dep', 'attributes': ['ADV'], 'link': 'dep', 'spans': [{'start': 199, 'end': 202}]}, {'word': '!', 'nodeType': 'dep', 'attributes': ['PUNCT'], 'link': 'dep', 'spans': [{'start': 202, 'end': 204}]}, {'word': 'think', 'nodeType': 'dep', 'attributes': ['VERB'], 'link': 'dep', 'spans': [{'start': 208, 'end': 214}], 'children': [{'word': '!', 'nodeType': 'dep', 'attributes': ['PUNCT'], 'link': 'dep', 'spans': [{'start': 204, 'end': 206}]}, {'word': 'I', 'nodeType': 'dep', 'attributes': ['PRON'], 'link': 'dep', 'spans': [{'start': 206, 'end': 208}]}, {'word': 'so', 'nodeType': 'dep', 'attributes': ['ADV'], 'link': 'dep', 'spans': [{'start': 214, 'end': 217}]}, {'word': 'Please', 'nodeType': 'dep', 'attributes': ['INTJ'], 'link': 'dep', 'spans': [{'start': 219, 'end': 226}], 'children': [{'word': '!', 'nodeType': 'dep', 'attributes': ['PUNCT'], 'link': 'dep', 'spans': [{'start': 217, 'end': 219}]}, {'word': 'work', 'nodeType': 'dep', 'attributes': ['VERB'], 'link': 'dep', 'spans': [{'start': 226, 'end': 231}]}, {'word': '!', 'nodeType': 'punct', 'attributes': ['PUNCT'], 'link': 'punct', 'spans': [{'start': 231, 'end': 233}]}]}]}]}]}]}]}]},
# 'nodeTypeToStyle': {'root': ['color5', 'strong'], 'dep': ['color5', 'strong'], 'nsubj': ['color1'], 'nsubjpass': ['color1'], 'csubj': ['color1'], 'csubjpass': ['color1'], 'pobj': ['color2'], 'dobj': ['color2'], 'iobj': ['color2'], 'mark': ['color2'], 'pcomp': ['color2'], 'xcomp': ['color2'], 'ccomp': ['color2'], 'acomp': ['color2'], 'aux': ['color3'], 'cop': ['color3'], 'det': ['color3'], 'conj': ['color3'], 'cc': ['color3'], 'prep': ['color3'], 'number': ['color3'], 'possesive': ['color3'], 'poss': ['color3'], 'discourse': ['color3'], 'expletive': ['color3'], 'prt': ['color3'], 'advcl': ['color3'], 'mod': ['color4'], 'amod': ['color4'], 'tmod': ['color4'], 'quantmod': ['color4'], 'npadvmod': ['color4'], 'infmod': ['color4'], 'advmod': ['color4'], 'appos': ['color4'], 'nn': ['color4'], 'neg': ['color0'], 'punct': ['color0']}, 'linkToPosition': {'nsubj': 'left', 'nsubjpass': 'left', 'csubj': 'left', 'csubjpass': 'left', 'pobj': 'right', 'dobj': 'right', 'iobj': 'right', 'pcomp': 'right', 'xcomp': 'right', 'ccomp': 'right', 'acomp': 'right'}}}

in_dict = {'arc_loss': 0.3790525794029236, 'tag_loss': 0.5370485782623291, 'loss': 0.9161011576652527, 'words': ['If', 'I', 'bring', '10', 'dollars', 'tomorrow', ',', 'can', 'you', 'buy', 'me', 'lunch', '?'], 'pos': ['ADP', 'PRON', 'VERB', 'NUM', 'NOUN', 'NOUN', 'PUNCT', 'VERB', 'PRON', 'VERB', 'PRON', 'NOUN', 'PUNCT'], 'predicted_dependencies': ['mark', 'nsubj', 'advcl', 'dep', 'dobj', 'tmod', 'advmod', 'aux', 'nsubj', 'root', 'dobj', 'dep', 'discourse'], 'predicted_heads': [3, 3, 10, 5, 3, 3, 10, 10, 10, 0, 10, 11, 10], 'hierplane_tree': {'text': 'If I bring 10 dollars tomorrow , can you buy me lunch ?', 'root': {'word': 'buy', 'nodeType': 'root', 'attributes': ['VERB'], 'link': 'root', 'spans': [{'start': 41, 'end': 45}], 'children': [{'word': 'bring', 'nodeType': 'advcl', 'attributes': ['VERB'], 'link': 'advcl', 'spans': [{'start': 5, 'end': 11}], 'children': [{'word': 'If', 'nodeType': 'mark', 'attributes': ['ADP'], 'link': 'mark', 'spans': [{'start': 0, 'end': 3}]}, {'word': 'I', 'nodeType': 'nsubj', 'attributes': ['PRON'], 'link': 'nsubj', 'spans': [{'start': 3, 'end': 5}]}, {'word': 'dollars', 'nodeType': 'dobj', 'attributes': ['NOUN'], 'link': 'dobj', 'spans': [{'start': 14, 'end': 22}], 'children': [{'word': '10', 'nodeType': 'dep', 'attributes': ['NUM'], 'link': 'dep', 'spans': [{'start': 11, 'end': 14}]}]}, {'word': 'tomorrow', 'nodeType': 'tmod', 'attributes': ['NOUN'], 'link': 'tmod', 'spans': [{'start': 22, 'end': 31}]}]}, {'word': ',', 'nodeType': 'advmod', 'attributes': ['PUNCT'], 'link': 'advmod', 'spans': [{'start': 31, 'end': 33}]}, {'word': 'can', 'nodeType': 'aux', 'attributes': ['VERB'], 'link': 'aux', 'spans': [{'start': 33, 'end': 37}]}, {'word': 'you', 'nodeType': 'nsubj', 'attributes': ['PRON'], 'link': 'nsubj', 'spans': [{'start': 37, 'end': 41}]}, {'word': 'me', 'nodeType': 'dobj', 'attributes': ['PRON'], 'link': 'dobj', 'spans': [{'start': 45, 'end': 48}], 'children': [{'word': 'lunch', 'nodeType': 'dep', 'attributes': ['NOUN'], 'link': 'dep', 'spans': [{'start': 48, 'end': 54}]}]}, {'word': '?', 'nodeType': 'discourse', 'attributes': ['PUNCT'], 'link': 'discourse', 'spans': [{'start': 54, 'end': 56}]}]}, 'nodeTypeToStyle': {'root': ['color5', 'strong'], 'dep': ['color5', 'strong'], 'nsubj': ['color1'], 'nsubjpass': ['color1'], 'csubj': ['color1'], 'csubjpass': ['color1'], 'pobj': ['color2'], 'dobj': ['color2'], 'iobj': ['color2'], 'mark': ['color2'], 'pcomp': ['color2'], 'xcomp': ['color2'], 'ccomp': ['color2'], 'acomp': ['color2'], 'aux': ['color3'], 'cop': ['color3'], 'det': ['color3'], 'conj': ['color3'], 'cc': ['color3'], 'prep': ['color3'], 'number': ['color3'], 'possesive': ['color3'], 'poss': ['color3'], 'discourse': ['color3'], 'expletive': ['color3'], 'prt': ['color3'], 'advcl': ['color3'], 'mod': ['color4'], 'amod': ['color4'], 'tmod': ['color4'], 'quantmod': ['color4'], 'npadvmod': ['color4'], 'infmod': ['color4'], 'advmod': ['color4'], 'appos': ['color4'], 'nn': ['color4'], 'neg': ['color0'], 'punct': ['color0']}, 'linkToPosition': {'nsubj': 'left', 'nsubjpass': 'left', 'csubj': 'left', 'csubjpass': 'left', 'pobj': 'right', 'dobj': 'right', 'iobj': 'right', 'pcomp': 'right', 'xcomp': 'right', 'ccomp': 'right', 'acomp': 'right'}}}

visualizer(in_dict)