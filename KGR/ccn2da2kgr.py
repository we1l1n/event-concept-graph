import codecs
from uri import *
import dask.dataframe as dd
import pandas as pd
from gensim.models import KeyedVectors
from gensim.similarities.index import AnnoyIndexer

# vocab
wv = KeyedVectors.load_word2vec_format('glove.6B.300d.bin', binary=True)
annoy_index = AnnoyIndexer()
annoy_index.load('glove.6B.300d.index')
annoy_index.model = wv

vocab_disasters = dict(wv.most_similar(positive=['disaster'], topn=2500, indexer=annoy_index))
vocab_accdients = dict(wv.most_similar(positive=['accident'], topn=2500, indexer=annoy_index))
vocab = list(set(list(vocab_disasters.keys()) + list(vocab_accdients.keys())));print('len(vocab):{}'.format(len(vocab)))

# ccn filt by vocab
df_conceptnet = dd.read_csv('./ccnet/neo-en-noURL/conceptnet-rel-*.csv',names=['start','rel','end','weight'],encoding='utf-8')
df_conceptnet['start_label'] = df_conceptnet['start'].apply(uri_to_label)
df_conceptnet['end_label'] = df_conceptnet['end'].apply(uri_to_label)

df_da = dd.concat([df_conceptnet[df_conceptnet.start_label.isin(vocab)],df_conceptnet[df_conceptnet.end_label.isin(vocab)]])
df_da = df_da.compute();print('len(df_da):{}'.format(len(df_da)))
df_da[['start','rel','end','weight']].to_csv('conceptnet_da_kgr/df-da.csv',index=False,header=False,encoding='utf-8')

# kb filt by rel
rel_count = df_da.groupby(['rel']).size()
rel_count.sort_values(ascending=False)
evidence_rel = list(rel_count.sort_values(ascending=False)[:-5].keys())
df_kgr = df_da[df_da.rel.isin(evidence_rel)];print('len(df_kgr):{}'.format(len(df_kgr)))
kgr_inv = [{'start':row[1]['end'],'rel':row[1]['rel']+'_inv','end':row[1]['start'],'weight':row[1]['weight']} for row in df_kgr.iterrows()]
df_kgr_inv = dd.from_pandas(pd.DataFrame.from_records(kgr_inv),npartitions=1)
df_kgr_all = dd.concat([df_kgr,df_kgr_inv],interleave_partitions=True).compute()
df_kgr_all[['start','end','rel']].to_csv('conceptnet_da_kgr/kb_env_rl.txt',sep='\t',index=False,header=False,encoding='utf-8')
print('len(df_kgr_all):{}'.format(len(df_kgr_all)))

# embeddings train files
entity = list(set(list(df_kgr_all.start.unique())+list(df_kgr_all.end.unique())));print('len(entity):{}'.format(len(entity)))
entity2id = dict([(x,i) for i,x in enumerate(entity)])
rel = list(df_kgr_all.rel.unique());print('len(rel):{}'.format(len(rel)))
relation2id = dict([(x,i) for i,x in enumerate(rel)])
train = []
for row in df_kgr_all.iterrows():
    train.append((entity2id[row[1]['start']],entity2id[row[1]['end']],relation2id[row[1]['rel']]))
print('len(train):{}'.format(len(train)))

f_entity = codecs.open('conceptnet_da_kgr/entity2id.txt','a+',encoding='utf-8')
f_relation = codecs.open('conceptnet_da_kgr/relation2id.txt','a+',encoding='utf-8')
f_train = codecs.open('conceptnet_da_kgr/train2id.txt','a+',encoding='utf-8')

f_entity.write(str(len(entity))+'\n')
f_relation.write(str(len(rel))+'\n')
f_train.write(str(len(train))+'\n')

[f_entity.write(k+'\t'+str(v)+'\n') for k,v in entity2id.items()]
[f_relation.write(k+'\t'+str(v)+'\n') for k,v in relation2id.items()]
[f_train.write(str(t[0])+'\t'+str(t[1])+'\t'+str(t[2])+'\n') for t in train]

# train_pos for task
task_rel = '/r/Causes'
train_pos = pd.read_csv('conceptnet_da_kgr/kb_env_rl.txt',sep='\t',names=['start','tail','rel'])
train_pos_ = train_pos[train_pos.rel == '/r/Causes']
train_pos_.to_csv('conceptnet_da_kgr/train_pos',sep='\t',header=False,index=False,encoding='utf-8')