import codecs,string
import numpy as np
import pandas as pd
from collections import Counter

# triples
triples = pd.read_csv('./triples-0.csv',index_col=None)
triples['triple'] = triples['head']+';'+triples['rel']+';'+triples['tail']
triple_count = triples.groupby('triple').size()
#triple_count.value_counts()[:10]

# count for new dataframe
triples_count = pd.DataFrame()
triples_count['triple'] = triple_count.keys()
triples_count['triple_count'] = triple_count.values
triples_count['start'] = triples_count.triple.apply(lambda x:x.split(';')[0])
triples_count['rel'] = triples_count.triple.apply(lambda x:x.split(';')[1])
triples_count['end'] = triples_count.triple.apply(lambda x:x.split(';')[2])
triples_count['weight'] = triples_count.triple_count.apply(lambda x:x/len(triples_count))

# vocab
vocabs = list(triples_count.start.map(lambda x:x.split('/')[-1])) + list(triples_count.end.map(lambda x:x.split('/')[-1]))
vocabs_count = Counter(vocabs)

# triples_count_filter
punct_re = '[!"#$%&\'()*+,-.:;<=>?@【[\\] ^`{|}~0-9]'
punct_filter = (triples_count.start.str.findall(punct_re).map(lambda x:len(x))+\
                triples_count.end.str.findall(punct_re).map(lambda x:len(x))).map(lambda x:False if x>0 else True)
triples_count_filter = triples_count[triples_count.triple_count > 1][punct_filter][triples_count.rel.str.startswith('/r/tweet')]

# rel filter
rel_count = triples_count_filter.groupby('rel').size()
evidence_rel = list(rel_count[rel_count > 100].keys())
triples_rel_filter = triples_count_filter[triples_count_filter.rel.isin(evidence_rel)]
triples_rel_filter[['start','rel','end','weight']].to_csv('df-tweet.csv',index=False)

# kb filt by rel
df_kgr = triples_rel_filter;print('len(df_kgr):{}'.format(len(df_kgr)))
kgr_inv = [{'start':row[1]['end'],'rel':row[1]['rel']+'_inv','end':row[1]['start'],'weight':row[1]['weight']} for row in df_kgr.iterrows()]
df_kgr_inv = pd.DataFrame.from_records(kgr_inv)
df_kgr_all = pd.concat([df_kgr,df_kgr_inv])
df_kgr_all[['start','end','rel']].to_csv('tweet_da_kgr/kb_env_rl.txt',sep='\t',index=False,header=False,encoding='utf-8')
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

f_entity = codecs.open('tweet_da_kgr/entity2id.txt','a+',encoding='utf-8')
f_relation = codecs.open('tweet_da_kgr/relation2id.txt','a+',encoding='utf-8')
f_train = codecs.open('tweet_da_kgr/train2id.txt','a+',encoding='utf-8')

f_entity.write(str(len(entity))+'\n')
f_relation.write(str(len(rel))+'\n')
f_train.write(str(len(train))+'\n')

[f_entity.write(k+'\t'+str(v)+'\n') for k,v in entity2id.items()]
[f_relation.write(k+'\t'+str(v)+'\n') for k,v in relation2id.items()]
[f_train.write(str(t[0])+'\t'+str(t[1])+'\t'+str(t[2])+'\n') for t in train]

# train_pos for task
task_rel = '/r/tweet/open/cause'
train_pos = pd.read_csv('tweet_da_kgr/kb_env_rl.txt',sep='\t',names=['start','tail','rel'])
train_pos_ = train_pos[train_pos.rel == task_rel]
train_pos_.to_csv('tweet_da_kgr/train_pos',sep='\t',header=False,index=False,encoding='utf-8')