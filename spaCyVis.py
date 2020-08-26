import cairosvg
import spacy
from spacy import displacy
from PyPDF2 import PdfFileMerger, PdfFileReader
def parser_to_input(dict1):
    """A program that will take a dictionary of the form of output and convert it to a form that can be given to the rendering function of displacy.
    This must include the words as separate dictionaries and the arcs that describe the start and ends of the words. Each arc will start at a certain 
    word (itself) and will end at its predicted head given in the input dictionary. Must account for the fact that the heads start counting from 1 since 
    0 is the root dependency."""
    input_dict = {}
    input_dict["words"] = text_format(dict1) #Format the text to be a list of dicts. 
    input_dict["arcs"] = getArcs(dict1) #Get the correct arcs 
    options = {"color": "red"} #Set options for style of tree
    html = displacy.render(input_dict, style = "dep", manual = True, options = options, jupyter = False) #Render the tree using displacy. Outputs HTML
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
    """Function that takes in a dependency dictionary and outputs the respective arcs that are in the dependency tree. Uses the displacy format to get
    a manual output. The output of this function is a list of dictionaries, each dictionary describes an arc between two words.
    """
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
    """Format the words of the sentence for manual rendering from displacy. This is a list of dictionaries, 
    one dictionary for each word of text and each word has a label with its POS."""
    words = dict1["words"]
    tags = dict1["pos"]
    text = []
    for i in range(len(words)):
        text_dict = {}
        text_dict["text"] = words[i]
        text_dict["tag"] = tags[i]
        text.append(text_dict)
    return text

def out_pdf(dict1, name):
    """Takes in a dictionary that maps an HTML string to a prediction dictionary and gives a PDF that contains the 
    displacy visual of a dependency for each sentence. This PDF will have one tree per page."""
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
    merger.write(name)
# out_pdf(dict2, "FullOutput.pdf")