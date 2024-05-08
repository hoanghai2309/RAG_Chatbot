import nltk
from nltk.parse import ChartParser
from nltk.parse.generate import generate, demo_grammar
from nltk import CFG
import shutil


# Task 2.2: Generate sentences
def generate_sentences(grammar_file, output_file):
    with open(grammar_file, 'r') as file:
        grammar_text = file.read()
    grammar = CFG.fromstring(grammar_text)
    with open(output_file, 'w') as file:
        for sentence in generate(grammar, n=1000):  # limit to 1000 sentences
            file.write(' '.join(sentence) + '\n')

# Task 2.3: Parse sentences
def parse_sentences(grammar_file, input_file, output_file):
    with open(grammar_file, 'r') as file:
        grammar_text = file.read()
    grammar = nltk.CFG.fromstring(grammar_text)
    parser = ChartParser(grammar)
    with open(input_file, 'r') as file:
        sentences = file.readlines()
    with open(output_file, 'w') as file:
        for sentence in sentences:
            try:
                trees = parser.parse(sentence.split())
                if trees:
                    for tree in trees:
                        file.write(str(tree) + '\n')
                else:
                    file.write('()\n')
            except Exception as e:
                file.write('()\n')
def test(**kwargs):
    shutil.copyfile('/src/demo_chatbot.mp4', '/nlp/output/demo_chatbot.mp4')
    shutil.copyfile('/src/part1/grammar.cfg', '/nlp/output/grammar.txt')
    # Generate sentences
    generate_sentences('/src/part1/grammar.cfg', '/nlp/output/samples.txt')
    # Parse sentences
    parse_sentences('/src/part1/grammar.cfg', '/nlp/input/samples.txt', '/nlp/output/parse-results.txt')