# bert-serving-start -model_dir E:\课件\nlp\bert_model\uncased_L-2_H-768_A-12

from bert_serving.client import BertClient
bc = BertClient()
vec = bc.encode(['First do it', 'then do it right', 'then do it better'])

print(vec.shape)  #(3,768)

print(vec[0])