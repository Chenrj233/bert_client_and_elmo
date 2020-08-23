from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import logging
import re
import nltk
import gensim
import pickle

import numpy as np
import pandas as pd

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from collections import defaultdict
from sklearn.model_selection import train_test_split

from bert_serving.client import BertClient
bc = BertClient()
vec = bc.encode(['First do it', 'then do it right', 'then do it better'])


print(vec.shape)  #(3,768)

print(vec[0])

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
    remove_stopwords = False #是否删除停用词
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
    w2v = {}
    word_vecs = bc.encode([word for word in diction])
    i = 0
    for word in diction:
        vec = word_vecs[i]
        w2v[word] = vec
        i = i + 1

    print(len(w2v))  #101398
    print(w2v['hello'])
    k = len(w2v['hello'])
    print(max_l)   #2297
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

    print(W.shape)  #(101400, 768)
    print(len(word_idx_map))  #101398
    print(W[3])
    print(W[4])
    exit()
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
    print(len(X_train)) #25000
    print(len(X_test))  #25000
    print(len(X_train[0]))  #437


    pickle_file = os.path.join('pickle', 'imdb_train_test_bert.pickle')
    pickle.dump([W, word_idx_map, max_l, X_train, Y_train, X_test], open(pickle_file, 'wb'))
    logging.info('dataset created!')