import codecs,json
from collections import Counter
import numpy as np
import pandas as pd
import dask.dataframe as dd
from uri import *

# tweet-triples
tweet = pd.read_csv('./tweet-triples.csv',index_col=None);print('len(tweet-triples):',len(tweet))

# count triple for filter
tweet['triple'] = tweet['head']+';'+tweet['rel']+';'+tweet['tail']
tweet_triple_count = tweet.groupby('triple').size()
tweet_new = pd.DataFrame()
tweet_new['triple'] = tweet_triple_count.keys()
tweet_new['triple_count'] = tweet_triple_count.values
tweet_new['start'] = tweet_new.triple.apply(lambda x:x.split(';')[0])
tweet_new['rel'] = tweet_new.triple.apply(lambda x:x.split(';')[1])
tweet_new['end'] = tweet_new.triple.apply(lambda x:x.split(';')[2])
tweet_new['weight'] = tweet_new.triple_count.apply(lambda x:x/len(tweet_new))
tweet_new['start_label'] = tweet_new['start'].apply(uri_to_label)
tweet_new['end_label'] = tweet_new['end'].apply(uri_to_label)
tweet_new['start_end_label'] = tweet_new['start_label']+tweet_new['end_label']

# char filter
invalid = set('[!"#$%&\'()*+,-.:;<=>?@®【[\\] ^`{|}~0123456789]')
def is_valid_phrase(s):
    return False if set(s)&invalid else True
tweet_new = tweet_new[tweet_new.start_end_label.apply(lambda x:is_valid_phrase(x))]\
                    [tweet_new.start != '/c/en/']\
                    [tweet_new.rel.str.startswith('/r/tweet')]
print('len(tweet_new):',len(tweet_new))

# vocab 
vocabs = list(tweet_new.start.map(lambda x:x.split('/')[-1])) + list(tweet_new.end.map(lambda x:x.split('/')[-1]))
vocabs_count = Counter(vocabs);print('len(vocabs):',len(vocabs_count.keys()))
with codecs.open('./tweet-vocabs.json','a+',encoding='utf-8') as f:
    json.dump(vocabs_count,f,indent=2)

# ccn
ccn = pd.read_csv('./ccn-triples.csv',names=['start','rel','end','weight'],encoding='utf-8');print('len(ccn-triples):',len(ccn))
ccn['start_label'] = ccn['start'].apply(uri_to_label)
ccn['end_label'] = ccn['end'].apply(uri_to_label)
# vocab filter
vocab = set(vocabs_count.keys())
ccn_new = ccn[ccn.start_label.apply(lambda x:x in vocab)]\
            [ccn.end_label.apply(lambda x:x in vocab)]
print('len(ccn_new):',len(ccn_new))

# merge
merge  = pd.concat([tweet_new[['start_label','rel','end_label','weight']],\
                ccn_new[['start_label','rel','end_label','weight']]],ignore_index=True)
print('len(merge):',len(merge))

# rel count filter
rel_count = merge.groupby(['rel']).size()
#rel_count.sort_values(ascending=False)
min_rel_count = 25
max_rel_count = 100000
# %matplotlib inline
# rel_count.value_counts().plot()
evidence_rel = list(rel_count[rel_count>min_rel_count]\
                             [rel_count<max_rel_count].sort_values(ascending=False).keys())

# df_kgr_all
merge.rename(index=str, columns={'start_label':'start','end_label':'end'},inplace=True)
df_kgr = merge[merge.rel.isin(evidence_rel)];print('len(df_kgr):{}'.format(len(df_kgr)))
kgr_inv = [{'start':row[1]['end'],'rel':row[1]['rel']+'_inv','end':row[1]['start'],'weight':row[1]['weight']} for row in df_kgr.iterrows()]
df_kgr_inv = pd.DataFrame.from_records(kgr_inv)
df_kgr_all = pd.concat([df_kgr,df_kgr_inv],ignore_index=True)
df_kgr_all[['start','end','rel']].to_csv('./kb_env_rl.txt',sep='\t',index=False,header=False,encoding='utf-8')
print('len(df_kgr_all):{}'.format(len(df_kgr_all)))

# X2id
entity = list(set(list(df_kgr_all.start.unique())+list(df_kgr_all.end.unique())));print('len(entity):{}'.format(len(entity)))
entity2id = dict([(x,i) for i,x in enumerate(entity)])
rel = list(df_kgr_all.rel.unique());print('len(rel):{}'.format(len(rel)))
relation2id = dict([(x,i) for i,x in enumerate(rel)])
train = []
for row in df_kgr_all.iterrows():
    train.append((entity2id[row[1]['start']],entity2id[row[1]['end']],relation2id[row[1]['rel']]))
print('len(train):{}'.format(len(train)))

f_entity = codecs.open('./entity2id.txt','a+',encoding='utf-8')
f_relation = codecs.open('./relation2id.txt','a+',encoding='utf-8')
f_train = codecs.open('./train2id.txt','a+',encoding='utf-8')

f_entity.write(str(len(entity))+'\n')
f_relation.write(str(len(rel))+'\n')
f_train.write(str(len(train))+'\n')

[f_entity.write(k+'\t'+str(v)+'\n') for k,v in entity2id.items()]
[f_relation.write(k+'\t'+str(v)+'\n') for k,v in relation2id.items()]
[f_train.write(str(t[0])+'\t'+str(t[1])+'\t'+str(t[2])+'\n') for t in train]

f_entity.close()
f_relation.close()
f_train.close()

# task
task_rels = ['/r/tweet/open/cause','/r/tweet/open/kill',\
            '/r/tweet/open/hit','/r/tweet/IsA','/r/Causes',\
            '/r/IsA','/r/AtLocation','/r/HasSubevent']

def rel_filter(df_kgr_all,task_rel):
    train_pos = df_kgr_all[df_kgr_all.rel == task_rel];print('len(train_pos):',task_rel,len(train_pos))
    rel_path = task_rel.replace('/','_')
    train_pos[['start','end','rel']].to_csv(rel_path+'_train_pos',sep='\t',header=False,index=False,encoding='utf-8')

[rel_filter(df_kgr_all,task_rel) for task_rel in task_rels]