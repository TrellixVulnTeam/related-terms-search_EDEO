"""
Intelligent search through text to find nouns, verbs, and adjectives associated with a keyword input.
@author Gabriel Hooks
@date 2021-02-04
"""

import bs4 as bs
import urllib.request
import urllib.error
import re
import nltk
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import spacy


def get_url():
    return input("Enter URL: ")


def get_key_terms():
    kts = []
    while True:
        kw = input("Keyword (enter 'done' to stop entering keywords): ").lower()
        if kw == "done":
            break
        else:
            kts.append(kw)
    return kts


try:
    scraped_data = urllib.request.urlopen(get_url())
except urllib.error.HTTPError as e:
    print("HTTP Error {} while trying to open {}".format(e.code, e.url))
    exit(1)

# scraped_data may be undefined, but if the try above fails, this will never be executed.
article = scraped_data.read()

parsed_article = bs.BeautifulSoup(article, 'lxml')

paragraphs = parsed_article.find_all('p')

article_text = ""

for p in paragraphs:
    article_text += p.text

# Cleaning the text
processed_article = article_text.lower()
processed_article = re.sub('[^a-zA-Z]', ' ', processed_article)
processed_article = re.sub(r'\s+', ' ', processed_article)

# Preparing the dataset
all_sentences = nltk.sent_tokenize(processed_article)

all_words = [nltk.word_tokenize(sent) for sent in all_sentences]

# Load Spacy's English model to remove stopwords alongside NLTKs stopwords
en_model = spacy.load('en_core_web_sm')
spacy_stopwords = en_model.Defaults.stop_words

# Remove stopwords
for i in range(len(all_words)):
    all_words[i] = [w for w in all_words[i] if w not in stopwords.words('english') and w not in spacy_stopwords]

# Get input for positive and negative keywords
print("Enter positive keywords...")
pos_terms = get_key_terms()
print("Enter negative keywords...")
neg_terms = get_key_terms()

# Make word2vec object converting all words within all_words that occur more than once.
word2vec = Word2Vec(all_words, min_count=2)
vocab = word2vec.wv.vocab

# Lists for final top results
top_nouns = []
top_verbs = []
top_adj = []

# Length of each top result list
top_w = 12

for i in range(len(pos_terms)):
    try:
        # Get 100 similar words using pos_terms and neg_terms
        sim_words = word2vec.wv.most_similar(positive=pos_terms[i], negative=neg_terms, topn=100)

        print("TERM={}:".format(pos_terms[i]))
        for j in range(len(sim_words)):
            # Get the POS tag from the current word from sim_words
            pos = nltk.pos_tag(nltk.word_tokenize(sim_words[j][0]))
            pos_tag = pos[0][1]

            print("{} ({}), {}".format(sim_words[j][0], pos_tag, sim_words[j][1]))

            # Get top results for each list.
            if pos_tag in ('NN', 'NNP', 'NNS') and len(top_nouns) < top_w:
                top_nouns.append((sim_words[j][0], pos_tag))
            elif pos_tag in ('VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ') and len(top_verbs) < top_w:
                top_verbs.append((sim_words[j][0], pos_tag))
            elif pos_tag in ('JJ', 'JJR', 'JJS') and len(top_adj) < top_w:
                top_adj.append((sim_words[j][0], pos_tag))

    except KeyError as e:
        print("ERROR! One or more keywords were not recognized as a vocabulary word! {}".format(e.args))
        exit(1)


print("\nTOP RESULTS\n")
print("NOUNS: {}\nVERBS: {}\nADJECTIVES: {}".format(top_nouns, top_verbs, top_adj))

# TODO: Export top results data to spreadsheet
#   e.g [KEYWORD] | [nouns] | [verbs] | [adjectives] | [similarity value]

# TODO: Input list of URLs, execute code for each URL and export results into spreadsheet

# TODO: Make better input functions. Do all inputs on one line separated by commas


# TODO: DONE Get top 6 Nouns, verbs, adjectives each
# TODO: DONE Search via multiple keywords
# TODO: DONE Refine results to only nouns, verbs, and adjectives
# TODO: Prevent 'KeyError: "word 'x' not in vocabulary'
# TODO: MOSTLY Make more accurate results
# TODO: MAYBE Scrape any website, not just wikipedia
# TODO: Insert and organize results into excel doc
