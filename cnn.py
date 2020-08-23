from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import logging
import pickle
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# use gpu


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.InteractiveSession(config=config)

# maxlen = 56
batch_size = 50
nb_epoch = 10
hidden_dim = 120

kernel_size = 3
nb_filter = 60

test = pd.read_csv("./corpus/imdb/testData.tsv", header=0,
                   delimiter="\t", quoting=3)


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))
    logging.info('loading data...')
    model_name = "elmo" #bert or elmo
    pickle_file = os.path.join('pickle', 'imdb_train_test_' + model_name +'.pickle')
    W, word_idx_map, max_l, X_train, Y_train, X_test = pickle.load(open(pickle_file, 'rb'))
    logging.info('data loaded!')
    x_train, x_dev, y_train, y_dev = train_test_split(X_train, Y_train, test_size=0.2, random_state=1)

    x_train = keras.preprocessing.sequence.pad_sequences(np.array(x_train), maxlen=max_l)
    x_dev = keras.preprocessing.sequence.pad_sequences(np.array(x_dev), maxlen=max_l)
    X_test = keras.preprocessing.sequence.pad_sequences(np.array(X_test), maxlen=max_l)

    y_train = keras.utils.to_categorical(np.array(y_train))
    y_dev = keras.utils.to_categorical(np.array(y_dev))
    train_len = len(X_train)  #总train数量
    max_features = W.shape[0]
    num_features = W.shape[1]

    print(x_train[0])
    print(x_train.shape)
    print(y_train.shape)
    print(y_train[:10])
    print(len(x_dev))
    print(y_dev.shape)
    print("X_test ", X_test.shape)
    # Keras Model
    #exit()
    sequence = keras.layers.Input(shape=(max_l,), dtype='int32')

    embedded = keras.layers.Embedding(input_dim=max_features, output_dim=num_features, input_length=max_l,
                                    weights=[W], trainable=False)(sequence)

    embedded = keras.layers.Dropout(0.25)(embedded)

    # convolutional layer
    convolution = keras.layers.Convolution1D(filters=nb_filter,
                                             kernel_size=kernel_size,
                                             padding='valid',
                                             activation='relu',
                                             strides=1
                                             )(embedded)
    maxpooling = keras.layers.MaxPooling1D(pool_size=2)(convolution)
    maxpooling = keras.layers.Flatten()(maxpooling)

    dense = keras.layers.Dense(70)(maxpooling)
    dense = keras.layers.Dropout(0.25)(dense)
    dense = keras.layers.Activation('relu')(dense)

    output = keras.layers.Dense(2, activation='softmax')(dense)
    model = keras.Model(inputs=sequence, outputs=output)
    print(x_train.shape)
    print((y_train.shape))
    print(X_test.shape)

    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    model.fit(x_train, y_train, validation_data=[x_dev, y_dev], batch_size=batch_size, epochs=nb_epoch)
    y_pred = model.predict(X_test, batch_size=batch_size)
    y_pred = np.argmax(y_pred, axis=1)

    result_output = pd.DataFrame(data={"id": test["id"], "sentiment": y_pred})


    result_output.to_csv("./result/cnn_" + model_name +".csv", index=False, quoting=3)
