import codecs,json
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

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
annoy_index_ent = AnnoyIndexer(wv_ent,200)
annoy_index_ent.save('./entity2vec.index')
# rel
wv_rel = KeyedVectors.load_word2vec_format('./relation2vec.txt', binary=False)
wv_rel.save_word2vec_format('./relation2vec.bin',binary=True)
wv_rel = KeyedVectors.load_word2vec_format('./relation2vec.bin',binary=True)
annoy_index_rel = AnnoyIndexer(wv_rel,200)
annoy_index_rel.save('./relation2vec.index')

# tsne-plot
def tsne_vis(X,labels,name):
    tsne = TSNE(n_components=2).fit_transform(X)
    plt.figure(figsize=(50,50))
    for i, label in enumerate(labels):
        x, y = tsne[i,:]
        plt.scatter(x,y)
        plt.annotate(label, xy=(x, y), xytext=(30,15), textcoords='offset points',
                        ha='right', va='bottom')
    plt.savefig('tsne-'+name+'.png')

#tsne_vis(wv_ent.vectors,[label.replace('concept_','').replace('/c/en/','') \
#                                            for label in wv_ent.vocab.keys()],name='ent')
tsne_vis(wv_rel.vectors,[label.replace('concept_','').replace('/r/en/tweet/','') \
                                            for label in wv_rel.vocab.keys()],name='rel')

def kmeans_label(X,names,n_clusters=2):
    clusters = defaultdict(list)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    [clusters[label].append(x) for label,x in zip(kmeans.labels_,names)]
    return clusters

clusters = kmeans_label(wv_rel.vectors,[label.replace('concept_','').replace('/r/tweet/','') \
                                            for label in wv_rel.vocab.keys()],n_clusters=100)
clusters = {str(k):v for k,v in clusters.items()}
with codecs.open('./rel-clusters.json','a+',encoding='utf-8') as f:
    json.dump(clusters,f,indent=2)