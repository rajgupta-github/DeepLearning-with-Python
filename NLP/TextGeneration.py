# Text Generation with LSTM with Keras and Python
import spacy


def read_file(filepath):
    with open(filepath) as f:
        str_text = f.read()

    return str_text


read_file('moby_dick_four_chapters.txt')

# python -m spacy download en
nlp = spacy.load('en', disable=['parser', 'tagger', 'ner'])

nlp.max_length = 1198623


def separate_punc(doc_text):
    return [token.text.lower() for token in nlp(doc_text) if token.text not in '\n\n \n\n\n!"-#$%&()--.*+,-/:;<=>?@[' \
                                                                               '\\]^_`{|}~\t\n ']


d = read_file('melville-moby_dick.txt')
tokens = separate_punc(d)

print(tokens)

