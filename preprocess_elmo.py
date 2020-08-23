from __future__ import absolute_import
from __future__ import print_function
import os
import sys
import logging
import pickle
import re
import numpy as np
import pandas as pd

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from collections import defaultdict
from sklearn.model_selection import train_test_split

from allennlp.modules.elmo import Elmo, batch_to_ids

options_file = os.path.join('e:\\', '课件', 'nlp', 'bert_client', 'elmo_2x1024_128_2048cnn_1xhighway_options.json')
weight_file = os.path.join('e:\\', '课件', 'nlp', 'bert_client', "elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5")

# Compute two different representation for each token.
# Each representation is a linear weighted combination for the
# 3 layers in ELMo (i.e., charcnn, the outputs of the two BiLSTM))
elmo = Elmo(options_file, weight_file, 2, dropout=0)


# Read data from files
train = pd.read_csv("./corpus/imdb/labeledTrainData.tsv", header=0,
                    delimiter="\t", quoting=3)
test = pd.read_csv("./corpus/imdb/testData.tsv", header=0,
                   delimiter="\t", quoting=3)
unlabeled_train = pd.read_csv("./corpus/imdb/unlabeledTrainData.tsv", header=0,
                              delimiter="\t", quoting=3)


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    print('begin')
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ''.join(sys.argv))
    diction = set()
    print(diction)
    remove_stopwords = False
    for i in range(len(train['sentiment'])):
        review_text = BeautifulSoup(train['review'][i], 'lxml').get_text()
        review_text = re.sub("[^a-zA-Z]", " ", review_text)
        words = review_text.lower().split()
        if remove_stopwords:
            stops = set(stopwords.words('english'))
            words = [w for w in words if not w in stops]
        for j in range(len(words)):
            diction.add(words[j])
    print(len(diction))
    max_l = 0
    for i in range(len(test['review'])):
        review_text = BeautifulSoup(test['review'][i], 'lxml').get_text()
        review_text = re.sub("[^a-zA-Z]", " ", review_text)
        words = review_text.lower().split()
        max_l = max(max_l, len(words))
        if remove_stopwords:
            stops = set(stopwords.words('english'))
            words = [w for w in words if not w in stops]
        for j in range(len(words)):
            diction.add(words[j])
    print(len(diction))
    '''dictionary = []
    for word in diction:
        x = [word]
        dictionary.append(x)'''
    #print(dictionary)
    #exit()
    w2v = {}
    i = 0
    for word in diction:
        character_ids = batch_to_ids([[word]])
        embeddings = elmo(character_ids)
        w2v[word] = embeddings['elmo_representations'][0].data.numpy()[0][0]
        if i + 1 % 5000 == 0:
            print("complete ", i, " words")
        i = i + 1
    print(len(w2v))  # 101398
    print(w2v['hello'])
    k = len(w2v['hello'])
    print(max_l)  # 2297
    logging.info('vec create!')
    vocab_size = len(w2v)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size + 2, k), dtype=np.float32)
    W[0] = np.zeros((k,))
    W[1] = np.random.uniform(-0.25, 0.25, k)

    i = 2
    for word in w2v:
        W[i] = w2v[word]
        word_idx_map[word] = i
        i = i + 1

    print(W.shape)  # (101400, 768)
    print(len(word_idx_map))  # 101398
    X_train, X_test, Y_train = [], [], []
    for i in range(len(train['review'])):
        x = []
        review_text = BeautifulSoup(train['review'][i], 'lxml').get_text()
        review_text = re.sub("[^a-zA-Z]", " ", review_text)
        words = review_text.lower().split()
        if remove_stopwords:
            stops = set(stopwords.words('english'))
            words = [w for w in words if not w in stops]
        for word in words:
            if word in word_idx_map:
                x.append(word_idx_map[word])
            else:
                x.append(1)
        X_train.append(x)
        Y_train.append(train['sentiment'][i])

    for i in range(len(test['review'])):
        x = []
        review_text = BeautifulSoup(test['review'][i], 'lxml').get_text()
        review_text = re.sub("[^a-zA-Z]", " ", review_text)
        words = review_text.lower().split()
        if remove_stopwords:
            stops = set(stopwords.words('english'))
            words = [w for w in words if not w in stops]
        for word in words:
            if word in word_idx_map:
                x.append(word_idx_map[word])
            else:
                x.append(1)
        X_test.append(x)
    print(len(X_train))  # 25000
    print(len(X_test))  # 25000
    print(len(X_train[0]))  # 437

    # pickle_file = os.path.join('pickle', 'vader_movie_reviews_glove.pickle3')
    pickle_file = os.path.join('pickle', 'imdb_train_test_elmo.pickle')
    pickle.dump([W, word_idx_map, max_l, X_train, Y_train, X_test], open(pickle_file, 'wb'))
    logging.info('dataset created!')