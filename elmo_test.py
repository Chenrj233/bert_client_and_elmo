import os
import sys
import logging

from allennlp.modules.elmo import Elmo, batch_to_ids

options_file = os.path.join('e:\\', '课件', 'nlp', 'bert_client', 'elmo_2x1024_128_2048cnn_1xhighway_options.json')
weight_file = os.path.join('e:\\', '课件', 'nlp', 'bert_client', "elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5")

# Compute two different representation for each token.
# Each representation is a linear weighted combination for the
# 3 layers in ELMo (i.e., charcnn, the outputs of the two BiLSTM))
elmo = Elmo(options_file, weight_file, 2, dropout=0)

# use batch_to_ids to convert sentences to character ids
sentences = [['First', 'sentence', 'hello','.'], ['Another', '.']]
character_ids = batch_to_ids(sentences)
print(character_ids)
print("ids")
print(character_ids[0][0])
print(len(character_ids[0][0]))
print(len(character_ids[0]))
print(len(character_ids))
print("embeddings")
#embeddings['elmo_representations']  :sentences 对应的embedding   每个单词一个
embeddings = elmo(character_ids)
print(embeddings)
print(len(embeddings['elmo_representations'][0].data.numpy()[0][0]))  #256
print(embeddings['elmo_representations'][0].data.numpy()[0][0])

print("data")
# embeddings['elmo_representations'] is length two list of tensors.
# Each element contains one layer of ELMo representations with shape
# (2, 3, 1024).
#   2    - the batch size
#   3    - the sequence length of the batch
#   1024 - the length of each ELMo vector

print(embeddings['elmo_representations'][0].data.numpy())
# print(embeddings['elmo_representations']['tensor'].shape)