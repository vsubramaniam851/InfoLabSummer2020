import spacy
from spacy import displacy
from spacy import util
from spacy.pipeline import DependencyParser
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF
from PyPDF2 import PdfFileMerger, PdfFileReader
import cairosvg
#Spacy Model 1: "en_core_web_sm". This model has a Dep Parser with a LAS of 89.68% and a UAS of 91.56%.
nlp = spacy.load("en_core_web_sm")
doc = nlp("This is a simple sentence")
# print([(w.pos_, w.dep_) for w in doc])
#Spacy Model 2: "en_core_web_lg". This model performs slightly better with a LAS of 90.27% and a UAS of 92.41%. 
nlp2 = spacy.load("en_core_web_lg")


example_dict = {
    "words": [
        {"text": "This", "tag": "DT"},
        {"text": "is", "tag": "VBZ"},
        {"text": "a", "tag": "DT"},
        {"text": "sentence", "tag": "NN"}
    ],
    "arcs": [
        {"start": 0, "end": 1, "label": "nsubj", "dir": "left"},
        {"start": 2, "end": 3, "label": "det", "dir": "left"},
        {"start": 1, "end": 3, "label": "attr", "dir": "right"}
    ]
}

# print(displacy.render(example_dict, style = "dep", manual = True))
def parser_to_input(dict1):
    """A program that will take a dictionary of the form of output and convert it to a form that can be given to the rendering function of displacy.
    This must include the words as separate dictionaries and the arcs that describe the start and ends of the words. Each arc will start at a certain 
    word (itself) and will end at its predicted head given in the input dictionary. Must account for the fact that the heads start counting from 1 since 
    0 is the root dependency."""
    input_dict = {}
    input_dict["words"] = text_format(dict1)
    input_dict["arcs"] = getArcs(dict1)
    options = {"color": "red"}
    html = displacy.render(input_dict, style = "dep", manual = True, options = options, jupyter = False)
    return html
def getDeps(dict1):
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
    return heads

def getArcs(dict1):
    predicted_heads = dict1["predicted_heads"]
    arcs = []
    predicted_deps = dict1["predicted_dependencies"]
    for i in range(len(predicted_heads)):
        arc_dict = {}
        if predicted_heads[i] != 0:
            if i < predicted_heads[i]:
                arc_dict["start"] = i
                arc_dict["end"] = predicted_heads[i] - 1
                arc_dict["dir"] = "left"
            else:
                arc_dict["start"] = predicted_heads[i] - 1
                arc_dict["end"] = i
                arc_dict["dir"] = "right"
            arc_dict["label"] = predicted_deps[i]
            arcs.append(arc_dict)
    return arcs

def text_format(dict1):
    words = dict1["words"]
    tags = dict1["pos"]
    text = []
    for i in range(len(words)):
        text_dict = {}
        text_dict["text"] = words[i]
        text_dict["tag"] = tags[i]
        text.append(text_dict)
    return text



in_dict = {'arc_loss': 0.3790525794029236, 'tag_loss': 0.5370485782623291, 'loss': 0.9161011576652527, 
'words': ['If', 'I', 'bring', '10', 'dollars', 'tomorrow', ',', 'can', 'you', 'buy', 'me', 'lunch', '?'], 
'pos': ['ADP', 'PRON', 'VERB', 'NUM', 'NOUN', 'NOUN', 'PUNCT', 'VERB', 'PRON', 'VERB', 'PRON', 'NOUN', 'PUNCT'], 
'predicted_dependencies': ['mark', 'nsubj', 'advcl', 'dep', 'dobj', 'tmod', 'advmod', 'aux', 'nsubj', 'root', 'dobj', 'dep', 'discourse'], 
'predicted_heads': [3, 3, 10, 5, 3, 3, 10, 10, 10, 0, 10, 11, 10], 
'hierplane_tree': {'text': 'If I bring 10 dollars tomorrow , can you buy me lunch ?', 'root': {'word': 'buy', 'nodeType': 'root', 'attributes': ['VERB'], 'link': 'root', 'spans': [{'start': 41, 'end': 45}], 'children': [{'word': 'bring', 'nodeType': 'advcl', 'attributes': ['VERB'], 'link': 'advcl', 'spans': [{'start': 5, 'end': 11}], 'children': [{'word': 'If', 'nodeType': 'mark', 'attributes': ['ADP'], 'link': 'mark', 'spans': [{'start': 0, 'end': 3}]}, {'word': 'I', 'nodeType': 'nsubj', 'attributes': ['PRON'], 'link': 'nsubj', 'spans': [{'start': 3, 'end': 5}]}, {'word': 'dollars', 'nodeType': 'dobj', 'attributes': ['NOUN'], 'link': 'dobj', 'spans': [{'start': 14, 'end': 22}], 'children': [{'word': '10', 'nodeType': 'dep', 'attributes': ['NUM'], 'link': 'dep', 'spans': [{'start': 11, 'end': 14}]}]}, {'word': 'tomorrow', 'nodeType': 'tmod', 'attributes': ['NOUN'], 'link': 'tmod', 'spans': [{'start': 22, 'end': 31}]}]}, {'word': ',', 'nodeType': 'advmod', 'attributes': ['PUNCT'], 'link': 'advmod', 'spans': [{'start': 31, 'end': 33}]}, {'word': 'can', 'nodeType': 'aux', 'attributes': ['VERB'], 'link': 'aux', 'spans': [{'start': 33, 'end': 37}]}, {'word': 'you', 'nodeType': 'nsubj', 'attributes': ['PRON'], 'link': 'nsubj', 'spans': [{'start': 37, 'end': 41}]}, {'word': 'me', 'nodeType': 'dobj', 'attributes': ['PRON'], 'link': 'dobj', 'spans': [{'start': 45, 'end': 48}], 'children': [{'word': 'lunch', 'nodeType': 'dep', 'attributes': ['NOUN'], 'link': 'dep', 'spans': [{'start': 48, 'end': 54}]}]}, {'word': '?', 'nodeType': 'discourse', 'attributes': ['PUNCT'], 'link': 'discourse', 'spans': [{'start': 54, 'end': 56}]}]}, 'nodeTypeToStyle': {'root': ['color5', 'strong'], 'dep': ['color5', 'strong'], 'nsubj': ['color1'], 'nsubjpass': ['color1'], 'csubj': ['color1'], 'csubjpass': ['color1'], 'pobj': ['color2'], 'dobj': ['color2'], 'iobj': ['color2'], 'mark': ['color2'], 'pcomp': ['color2'], 'xcomp': ['color2'], 'ccomp': ['color2'], 'acomp': ['color2'], 'aux': ['color3'], 'cop': ['color3'], 'det': ['color3'], 'conj': ['color3'], 'cc': ['color3'], 'prep': ['color3'], 'number': ['color3'], 'possesive': ['color3'], 'poss': ['color3'], 'discourse': ['color3'], 'expletive': ['color3'], 'prt': ['color3'], 'advcl': ['color3'], 'mod': ['color4'], 'amod': ['color4'], 'tmod': ['color4'], 'quantmod': ['color4'], 'npadvmod': ['color4'], 'infmod': ['color4'], 'advmod': ['color4'], 'appos': ['color4'], 'nn': ['color4'], 'neg': ['color0'], 'punct': ['color0']}, 'linkToPosition': {'nsubj': 'left', 'nsubjpass': 'left', 'csubj': 'left', 'csubjpass': 'left', 'pobj': 'right', 'dobj': 'right', 'iobj': 'right', 'pcomp': 'right', 'xcomp': 'right', 'ccomp': 'right', 'acomp': 'right'}}}


in_dict2 = {'arc_loss': 0.03730561211705208, 'tag_loss': 0.13279448449611664, 'loss': 0.17010009288787842, 
'words': ['Hello', 'my', 'name', 'is', 'Vighnesh', '!'], 
'pos': ['INTJ', 'DET', 'NOUN', 'VERB', 'PROPN', 'PUNCT'], 
'predicted_dependencies': ['csubj', 'dep', 'dep', 'cop', 'root', 'punct'], 
'predicted_heads': [5, 3, 1, 5, 0, 5], 
'hierplane_tree': {'text': 'Hello my name is Vighnesh !', 'root': {'word': 'Vighnesh', 'nodeType': 'root', 'attributes': ['PROPN'], 'link': 'root', 'spans': [{'start': 17, 'end': 26}], 'children': [{'word': 'Hello', 'nodeType': 'csubj', 'attributes': ['INTJ'], 'link': 'csubj', 'spans': [{'start': 0, 'end': 6}], 'children': [{'word': 'name', 'nodeType': 'dep', 'attributes': ['NOUN'], 'link': 'dep', 'spans': [{'start': 9, 'end': 14}], 'children': [{'word': 'my', 'nodeType': 'dep', 'attributes': ['DET'], 'link': 'dep', 'spans': [{'start': 6, 'end': 9}]}]}]}, {'word': 'is', 'nodeType': 'cop', 'attributes': ['VERB'], 'link': 'cop', 'spans': [{'start': 14, 'end': 17}]}, {'word': '!', 'nodeType': 'punct', 'attributes': ['PUNCT'], 'link': 'punct', 'spans': [{'start': 26, 'end': 28}]}]}, 'nodeTypeToStyle': {'root': ['color5', 'strong'], 'dep': ['color5', 'strong'], 'nsubj': ['color1'], 'nsubjpass': ['color1'], 'csubj': ['color1'], 'csubjpass': ['color1'], 'pobj': ['color2'], 'dobj': ['color2'], 'iobj': ['color2'], 'mark': ['color2'], 'pcomp': ['color2'], 'xcomp': ['color2'], 'ccomp': ['color2'], 'acomp': ['color2'], 'aux': ['color3'], 'cop': ['color3'], 'det': ['color3'], 'conj': ['color3'], 'cc': ['color3'], 'prep': ['color3'], 'number': ['color3'], 'possesive': ['color3'], 'poss': ['color3'], 'discourse': ['color3'], 'expletive': ['color3'], 'prt': ['color3'], 'advcl': ['color3'], 'mod': ['color4'], 'amod': ['color4'], 'tmod': ['color4'], 'quantmod': ['color4'], 'npadvmod': ['color4'], 'infmod': ['color4'], 'advmod': ['color4'], 'appos': ['color4'], 'nn': ['color4'], 'neg': ['color0'], 'punct': ['color0']}, 'linkToPosition': {'nsubj': 'left', 'nsubjpass': 'left', 'csubj': 'left', 'csubjpass': 'left', 'pobj': 'right', 'dobj': 'right', 'iobj': 'right', 'pcomp': 'right', 'xcomp': 'right', 'ccomp': 'right', 'acomp': 'right'}}}



# out = text_format(in_dict)
# print(out)
# arcs = getArcs(in_dict)
# print(arcs)
out_html = parser_to_input(in_dict)
out_html2 = parser_to_input(in_dict2)
# print(out_html)
dict2 = {}
dict2[out_html] = in_dict
dict2[out_html2] = in_dict2

# print(out_html2)

# file_name = '-'.join(in_dict["words"])
# svg_file_name = file_name + '.svg'
# pdf_file_name = file_name + '.pdf'
# with open(svg_file_name, "w", encoding = "utf-8") as im:
#     im.write(out_html)
# im.close()
# cairosvg.svg2pdf(url = svg_file_name, write_to = pdf_file_name)
# drawing = svg2rlg(file_name)
# pdf_file_name = '-'.join(in_dict["words"]) + '.pdf'
# renderPDF.drawToFile(drawing, pdf_file_name)


def out_pdf(dict1):
    PDFs = []
    for key in dict1:
        # print(key)
        input_dict = dict1[key]
        file_name = '-'.join(input_dict["words"])
        svg_file_name = file_name + '.svg'
        with open(svg_file_name, "w", encoding = "utf-8") as im:
            im.write(key)
        im.close()
        pdf_file_name = file_name + '.pdf'
        cairosvg.svg2pdf(url = svg_file_name, write_to = pdf_file_name)
        PDFs.append(pdf_file_name)  
    merger = PdfFileMerger()
    for filename in PDFs:
        merger.append(PdfFileReader(filename, 'rb'))
    merger.write("FullOutput.pdf")
out_pdf(dict2)
