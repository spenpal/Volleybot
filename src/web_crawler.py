# Filename:     hw5_sxp170022.py
# Date:         3/7/21
# Author:       Sanjeev Penupala
# Email:        sanjeev.penupala@utdallas.edu
# Course:       CS 4395.0W1
# Copyright     2021, All Rights Reserved
#
# Description:
#
#       Create a knowledge base scraped from the web.
#       This knowledge base will be used to create a chatbot that can carry
#       on a limited conversation in a particular domain using the knowledge base,
#       as well as knowledge it learns from the user.
#

###########
# IMPORTS #
###########

import math
import os
import pickle
import re
import unicodedata

# Standard Library Imports
from collections import Counter, deque
from pathlib import Path
from pprint import pprint

import requests

# Third Party Library Imports
from bs4 import BeautifulSoup
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

###########
# GLOBALS #
###########

cwd = Path.cwd()  # current working directory
topic = "volleyball"  # topic to be web crawled for
starter_url = "https://en.wikipedia.org/wiki/Volleyball"  # starting url to web crawl
count = 40  # number of relevant urls to be retrived
term_count = 40  # number of top terms to find in documents
ordered_nums = iter(range(1, count + 1))  # to name files by number
top_10_terms = [
    "line",
    "ball",
    "morgan",
    "attack",
    "court",
    "point",
    "libero",
    "setter",
    "fivb",
    "championship",
]


#############
# FUNCTIONS #
#############


def relevant_url(topic, link):
    blacklist = [
        "google",
        "pinterest",
        "imdb",
        "facebook",
        "reddit",
        "twitter",
        "whatsapp",
        "pdf",
        "php",
        "jpg",
        "product",
        "shop",
        "mail",
        "video",
        "share",
    ]

    if not link.startswith("http"):
        return False
    if topic not in link:
        return False
    if any(b in link for b in blacklist):
        return False
    if "wikipedia" in link and "en.wikipedia" not in link:
        return False

    return True


def web_crawler(topic, starter_url, count):
    relevant_urls = set()
    urls = deque([starter_url])

    while len(relevant_urls) < count and urls:
        url = urls.popleft()

        # Check if URL is accessible and/or has content
        try:
            r = requests.get(url)
            if r.status_code != 200:
                continue
        except requests.exceptions.RequestException:
            continue

        soup = BeautifulSoup(r.text, "html.parser")

        for link in soup.find_all("a"):
            link_str = str(link.get("href")).lower()

            # Edit Scraped Link
            if link_str.startswith("/url?q="):
                link_str = link_str[7:]
                print("MOD:", link_str)
            if "&" in link_str:
                i = link_str.find("&")
                link_str = link_str[:i] if i != -1 else link_str

            # Check if relevant URL
            if not relevant_url(topic=topic, link=link_str):
                continue

            urls.append(link_str)

        relevant_urls.add(url)

    return relevant_urls


def web_scraper(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.text, "lxml")

    if not soup.title:
        return

    page = []
    for p in soup.find_all("p"):
        page.append(p.text)

    title = f"{topic}-{next(ordered_nums)}"
    file_path = cwd.joinpath(f"data/scraped/{title}.txt")

    with open(file_path, "w") as f:
        f.write("\n".join(page))


def text_prettify(filename):
    scraped_file_path = cwd.joinpath(f"data/scraped/{filename}")
    clean_file_path = cwd.joinpath(f"data/clean/{filename}")

    with open(scraped_file_path, "r") as f:
        text = f.read()
        text = re.sub(r"[\n\t]", " ", text)
        sents = sent_tokenize(text)

        new_sents = []
        for sent in sents:
            new_sents += re.split(r"\s{3,}", sent)
        new_sents = [sent for sent in new_sents if sent]

    with open(clean_file_path, "w") as f:
        f.write("\n".join(new_sents))


def important_terms(count, dir_path):
    def preprocessing(doc):
        doc = doc.lower()  # lower case text
        doc = re.sub(r"[^\w\s]", " ", doc)  # remove punctuation

        stop_words = stopwords.words("english")
        tokens = word_tokenize(doc)  # tokenize document
        tokens = [
            token for token in tokens if token not in stop_words
        ]  # remove stop words

        wnl = WordNetLemmatizer()
        tokens = [wnl.lemmatize(token) for token in tokens]  # lemmatize tokens
        tokens = [
            token for token in tokens if len(token) > 3
        ]  # remove words less than 4 letters

        return tokens

    def compute_tf(doc_tokens):
        word_counts = Counter(doc_tokens)
        return {word: (count / len(doc_tokens)) for word, count in word_counts.items()}

    def compute_idf(docs, bag_of_words):
        idfs = dict.fromkeys(bag_of_words, 0)
        N = len(docs)

        for word in idfs:
            for doc, tokens in docs.items():
                if word in tokens:
                    idfs[word] += 1

        for word, count in idfs.items():
            idfs[word] = math.log(N / count)

        return idfs

    def compute_tfidf(tf, idfs):
        tfidf = {}
        for word, val in tf.items():
            tfidf[word] = val * idfs[word]
        return tfidf

    # Create of dictionary of keys of documents and values of tokens of each document
    docs = {}
    for filename in sorted(os.listdir(dir_path)):
        file_path = os.path.join(dir_path, filename)
        if os.path.isfile(file_path):
            with open(file_path, "r") as f:
                text = f.read()
                tokens = preprocessing(text)
                docs[filename] = tokens

    # Get vocabulary from all documents
    bag_of_unique_words = set()
    for doc, tokens in docs.items():
        bag_of_unique_words |= set(tokens)

    # Compute term frequencies for each document
    tfs = {}
    for doc, tokens in docs.items():
        tfs[doc] = compute_tf(tokens)

    # Compute inverse document frequencies for entire vocabulary
    idfs = compute_idf(docs, bag_of_unique_words)

    # Compute TF-IDF for each word in each document
    tf_idfs = {}
    for doc, tf in tfs.items():
        tf_idfs[doc] = compute_tfidf(tf, idfs)

    # Find avg TF-IDF for each word from all documents
    avg_tf_idfs = dict.fromkeys(bag_of_unique_words, 0)
    for word in avg_tf_idfs:
        avd_tf_idf = sum(tf_idf.get(word, 0) for tf_idf in tf_idfs.values()) / len(
            tf_idfs
        )
        avg_tf_idfs[word] = round(avd_tf_idf, 4)

    # Return top TF-IDF terms
    top_terms = Counter(avg_tf_idfs).most_common(count)
    return top_terms


def create_knowledge_base(terms, dir_path):
    kb = {term: [] for term in terms}

    for filename in sorted(os.listdir(dir_path)):
        file_path = os.path.join(dir_path, filename)
        if os.path.isfile(file_path):
            with open(file_path, "r") as f:
                for sent in f.readlines():
                    sent = unicodedata.normalize("NFKD", sent)
                    sent = sent.strip()

                    for term in kb:
                        if term in sent.lower():
                            kb[term].append(sent)

    return kb


########
# MAIN #
########


def main():
    print("STATUS UPDATES:")
    # Create data directory and relevant sub directories
    url_file_path = cwd.joinpath("data/relevant_urls.txt")
    scraped_dir_path = cwd.joinpath("data/scraped")
    clean_dir_path = cwd.joinpath("data/clean")

    if os.path.isdir(cwd.joinpath("data")):
        print(
            '-> Knowledge base is already compiled. Delete "data/" directory for new compilation.'
        )
    else:
        print(f"-> Crawling web for links related to {topic}...")
        relevant_urls = web_crawler(
            topic=topic.lower(), starter_url=starter_url.lower(), count=count
        )
        print(f"-> Finished crawling the web!")

        os.makedirs(clean_dir_path, exist_ok=True)
        os.makedirs(scraped_dir_path, exist_ok=True)

        # Write all relevant urls to a file
        print("-> Wrote relevant urls to text file!")
        with open(url_file_path, "w") as f:
            f.write("\n".join(f"{i+1}. {url}" for i, url in enumerate(relevant_urls)))

        # Scrape all relevant urls and put them in their own files
        print("-> Scraping all pages from relevant urls...")
        for url in relevant_urls:
            web_scraper(url)
        print("-> Scraped!")

        # Clean all scraped data and put it in separate directory
        print("-> Cleaning all scraped pages...")
        for filename in os.listdir(scraped_dir_path):
            text_prettify(filename)
        print("-> Cleaned!")

    # Print top terms from all documents using tf-idf
    top_terms = important_terms(term_count, clean_dir_path)
    print()
    print(f"Top {term_count} Important Terms:")
    pprint(top_terms)

    # Create knowledge base from top 10 terms (chosen manually)
    # Knowledge Base Format:
    #     kb = {
    #     '<term>': ['fact or sentence', 'fact or sentence', ...],
    #     '<term2>': ['fact or sentence', 'fact or sentence', ...]
    #     }
    kb = create_knowledge_base(top_10_terms, clean_dir_path)
    kb_file_path = cwd.joinpath("data/knowledge_base.pickle")
    pickle.dump(kb, open(kb_file_path, "wb"))

    example_kb = {term: facts[:1] for term, facts in kb.items()}
    print()
    print("Example Structure of Knowledge Base:")
    pprint(example_kb)


if __name__ == "__main__":
    main()
