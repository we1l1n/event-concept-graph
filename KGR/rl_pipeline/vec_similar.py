import codecs,json
import numpy as np
from gensim.models import KeyedVectors
from gensim.similarities.index import AnnoyIndexer

# id,embeddings
entity2id = dict([(line.split()[0], int(line.split()[1])) for line in
      codecs.open('./entity2id.txt', 'r', encoding='utf-8') if len(line.split()) == 2])
relation2id = dict([(line.split()[0], int(line.split()[1])) for line in
            codecs.open('./relation2id.txt', 'r', encoding='utf-8') if len(line.split()) == 2])
embeddings = json.load(open('./E.vec.json', 'r'))
entity2vec = np.array(embeddings['ent_embeddings'])
relation2vec = np.array(embeddings['rel_embeddings'])

# check
assert entity2vec.shape[0] == len(entity2id);assert entity2vec.shape[1] == 100
assert relation2vec.shape[0] == len(relation2id);assert relation2vec.shape[1] == 100

# w2v
with codecs.open('./entity2vec.txt','a+',encoding='utf-8') as f:
    f.write(str(entity2vec.shape[0])+' '+str(entity2vec.shape[1])+'\n')
    for line in zip(entity2id.items(),entity2vec):
        f.write(line[0][0]+' '+' '.join([str(vec) for vec in line[1]])+'\n')
with codecs.open('./relation2vec.txt','a+',encoding='utf-8') as f:
    f.write(str(relation2vec.shape[0])+' '+str(relation2vec.shape[1])+'\n')
    for line in zip(relation2id.items(),relation2vec):
        f.write(line[0][0]+' '+' '.join([str(vec) for vec in line[1]])+'\n')

# word2vec bin
wv_ent = KeyedVectors.load_word2vec_format('./entity2vec.txt', binary=False)
wv_ent.save_word2vec_format('./entity2vec.bin',binary=True)
# annoy index
wv_ent = KeyedVectors.load_word2vec_format('./entity2vec.bin',binary=True)
annoy_index_ent = AnnoyIndexer(wv,200)
annoy_index_ent.save('./entity2vec.index')
# rel
wv_rel = KeyedVectors.load_word2vec_format('./relation2vec.txt', binary=False)
wv_rel.save_word2vec_format('./relation2vec.bin',binary=True)
wv_rel = KeyedVectors.load_word2vec_format('./relation2vec.bin',binary=True)
annoy_index_rel = AnnoyIndexer(wv,200)
annoy_index_rel.save('./relation2vec.index')