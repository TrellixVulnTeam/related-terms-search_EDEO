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
    """
    :return: List of URLs from user input separated by spaces.
    """
    return [i for i in input("Enter URLs (separated by a space): ").split()]


def get_pos_terms():
    """
    :return: List of positive contributor terms from user input separated by spaces.
    """
    return [i for i in input("Enter positive contributors (separated by a space): ").split()]


def get_neg_terms():
    """
    :return: List of negative contributor terms from user input separated by spaces.
    """
    return [i for i in input("Enter negative contributors (separated by a space): ").split()]


def parse_url(url):
    """
    Parse, process, and clean text data from a URL to be searched through.

    :param url: The URL to search through.
    :return: List of all non-stopwords within a text.
    """
    scraped_data = urllib.request.urlopen(url)

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

    words = [nltk.word_tokenize(sent) for sent in all_sentences]

    # Load Spacy's English model to remove stopwords alongside NLTKs stopwords
    en_model = spacy.load('en_core_web_sm')
    spacy_stopwords = en_model.Defaults.stop_words

    # Remove stopwords
    for i in range(len(words)):
        words[i] = [w for w in words[i] if w not in stopwords.words('english') and w not in spacy_stopwords]

    return words


def get_top_sim_words(text, top_w = 12):
    """
    Get the top 'top_w' nouns, verbs, and adjectives related to the positive contributors and negative contributors.

    :param text: The cleaned text to search through.
    :param top_w: The number of nouns, verbs, and adjectives, each, the function will attempt to find.
    :return: List of lists containing top nouns, verbs, adjectives.
    """
    # Get input for positive and negative keywords
    pos_terms = get_pos_terms()
    neg_terms = get_neg_terms()

    # Make word2vec object converting all words within all_words that occur more than once.
    word2vec = Word2Vec(text, min_count=2)
    # vocab = word2vec.wv.vocab

    # Lists for final top results
    top_nouns = []
    top_verbs = []
    top_adj = []

    for i in range(len(pos_terms)):
        try:
            # Get 100 similar words using pos_terms and neg_terms
            sim_words = word2vec.wv.most_similar(positive=pos_terms, negative=neg_terms, topn=100)

            # print("TERM={}:".format(pos_terms[i]))
            for j in range(len(sim_words)):
                # Get the POS tag from the current word from sim_words
                pos = nltk.pos_tag(nltk.word_tokenize(sim_words[j][0]))
                pos_tag = pos[0][1]

                # print("{} ({}), {}".format(sim_words[j][0], pos_tag, sim_words[j][1]))

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
    return [top_nouns, top_verbs, top_adj]


def main():
    urls = get_url()
    for u in urls:
        try:
            print(u)
            all_words = parse_url(u)
        except urllib.error.HTTPError as e:
            print("HTTP Error {} while trying to open {}".format(e.code, e.url))
            continue
        top_results = get_top_sim_words(all_words, 12)
        print("\nTOP RESULTS FOR {}".format(u))
        print("NOUNS: {}\nVERBS: {}\nADJECTIVES: {}\n".format(top_results[0], top_results[1], top_results[2]))


if __name__ == '__main__':
    main()
