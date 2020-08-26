from allennlp.predictors.predictor import Predictor
import allennlp_models.syntax.biaffine_dependency_parser
predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/biaffine-dependency-parser-ptb-2020.04.06.tar.gz")



prediction = predictor.predict(
  sentence="If I bring 10 dollars tomorrow, can you buy me lunch?"
)
tree = prediction['hierplane_tree']
root = tree['root']
def dependencies(dict1, return_dict):
    if 'children' not in dict1:
        return return_dict
    else:
        child_dict = dict1['children']
        new_word = dict1['word']
        return_dict[new_word] = []
        for child in child_dict:
            return_dict[new_word] = return_dict[new_word] + [child['word']]
            return_dict = dependencies(child, return_dict)
        return return_dict
deps = dependencies(root, {})

def depth(dict1, count, word):
    num = 0
    for key in dict1:
        if word not in dict1[key]:
            num = num + 1
        elif word in dict1[key]:
            out_word = key
    if num == len(dict1):
        return count
    else:
        count = count + 1
        return depth(dict1, count, out_word)

def left_dep(dict1, sentence):
    count = 0
    for key in dict1:
        value = dict1[key]
        for i in value:
            try:
                if sentence.index(i) < sentence.index(key):
                    count = count + 1
            except ValueError:
                pass
    return count
print(left_dep(deps, prediction["words"]))

def root(dict1):
    deps = dict1["predicted_dependencies"]
    for i in range(len(deps)):
        if deps[i] == "root":
            index = i
    words = dict1["words"]
    return words[index]
print(root(prediction)) 