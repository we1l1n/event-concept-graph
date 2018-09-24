
import codecs,json
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

from gensim.models import KeyedVectors
from gensim.similarities.index import AnnoyIndexer

wv_ent = KeyedVectors.load_word2vec_format('entity2vec.bin', binary=True)
annoy_index_ent = AnnoyIndexer()
annoy_index_ent.load('entity2vec.index')
annoy_index_ent.model = wv_ent

wv_rel = KeyedVectors.load_word2vec_format('relation2vec.bin', binary=True)
annoy_index_rel = AnnoyIndexer()
annoy_index_rel.load('relation2vec.index')
annoy_index_rel.model = wv_rel

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
tsne_vis(wv_rel.vectors,[label.replace('concept_','').replace('/r/tweet/','') \
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