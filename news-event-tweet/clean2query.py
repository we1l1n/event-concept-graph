# coding: utf-8
from datetime import datetime,timedelta
from collections import defaultdict,Counter
from pprint import pprint
from tqdm import tqdm
import re

import pymongo
from pymongo import InsertOne, DeleteMany, ReplaceOne, UpdateOne
from pymongo.errors import BulkWriteError
client = pymongo.MongoClient('localhost:27017')
db = client.tweet
#db.authenticate('admin','lixiepeng')

from nltk.corpus import stopwords
list_stopWords=list(set(stopwords.words('english')))

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import itertools

import spacy
nlp = spacy.load('en_core_web_md')

# from mongo
events = [e for e in db.current_event.find({},{'_id':1,'event.class':1,'event.date':1,'event.title':1,'event.description':1})]
# to dict
events = [{'id':e['_id'],'class':e['event']['class'],'date':e['event']['date'],'title':e['event']['title'],'description':e['event']['description']} for e in events]
# to dataframe
df_events = pd.DataFrame.from_records(events)

# unify class using class_code
def class_code(type_str):
    type_str = type_str.lower()
    if 'armed' in type_str or 'attack' in type_str or 'conflict' in type_str:
        return 1
    elif 'disaster' in type_str or 'accident' in type_str:
        return 2
    elif 'law' in type_str or 'crime' in type_str:
        return 3
    elif 'politic' in type_str or 'election' in type_str:
        return 4
    elif 'international' in type_str or 'relation' in type_str:
        return 5
    elif 'science' in type_str or 'technology' in type_str:
        return 6
    elif 'business' in type_str or 'econom' in type_str:
        return 7
    elif 'art' in type_str or 'culture' in type_str:
        return 8
    elif 'sport' in type_str:
        return 9
    elif 'health' in type_str or 'environment' in type_str:
        return 10
    else:
        return 0
# apply transform function
df_events['class_code'] = df_events['class'].apply(class_code)
# clean multi-description
df_events['des_num'] = df_events['description'].apply(lambda x:len(x.split('\n')))

'''
def split_des(description):
    description = description.split('\n')
    des_ = []
    for des in description:
        #des_.append({'media':re.findall('(?:\.\s|\)[\s]+|\.)\(([\w\s\.]+)\)',des),'abstract':re.sub('(?:\.\s|\)[\s]+|\.)\([\w\s\.]+\)','',des).strip()})
        #des = des.split(['. (','.('])
        des = re.split('\.[\s]{0,}\(',des)
        des_.append({'media':re.findall('([\w\s\.]+)\)',des[1]),'abstract':des[0]+'.'})
    return des_
'''

def description_clean(description):
    description = description.split('. (')[0]+'.'
    return description

df_events['des_clean'] = df_events['description'].apply(description_clean)

'''
def efitf(X):
    count = CountVectorizer(stop_words='english')
    X_train_count = count.fit_transform(X)
    tfidf = TfidfTransformer(use_idf=True,smooth_idf=True,sublinear_tf=True)
    X_train_tfidf = tfidf.fit_transform(X_train_count)
    tf_feature_names = count.get_feature_names()
    X_train_tfidf = [list(i) for i in list(X_train_tfidf.toarray())]
    EFITF = defaultdict(dict)
    for Type,values in enumerate(X_train_tfidf):
        for index,value in enumerate(values):
            if value > 0.0:
                EFITF[Type].update({tf_feature_names[index]:value}) 
    return EFITFX = []X = df_events['des_clean'].tolist()EFITF = efitf(X)
'''
def class_similarity(class_text,span):
    return nlp(class_text).similarity(nlp(span))

def get_query(doc,class_text,doc_index,doc_date):
    doc_date = datetime.strptime(doc_date,'%Y-%m-%d')
    date_0 = doc_date.strftime('%Y-%m-%d')
    date_0_ = (doc_date+timedelta(days=-3)).strftime('%Y-%m-%d')
    date_1 = (doc_date+timedelta(days=1)).strftime('%Y-%m-%d')
    date_1_ = date_0
    doc = nlp(doc)
    kws = []
    for i in doc.ents:
        kws.append(i.text)
    triggers = []
    for token in doc:
        if not token.is_stop and token.tag_.startswith('V'):
            if token.text.lower() not in list_stopWords:
                triggers.append((token.text,token.tag_,str(class_similarity(class_text,token.text))))
    triggers = sorted(triggers,key=lambda x:x[2],reverse=True)[:3]
    for i in triggers:
        kws.append(i[0])
    noun_chunks = []
    for i in doc.noun_chunks:
        noun_chunks.append((i.text,str(class_similarity(class_text,i.text))))
    try:
        kws.append(sorted(noun_chunks,key=lambda x:x[1],reverse=True)[0][0].split(' ')[-1])
    except:
        pass
    kws = [w for w in kws if not w in list_stopWords]
    kws = list(set(kws))
    query = [i for i in itertools.combinations(kws,2)]
    query = ['"'+i[0]+'"'+' '+'"'+i[1]+'"'+' '+'since:'+date_0_+' '+'until:'+date_0 for i in query]+['"'+i[0]+'"'+' '+'"'+i[1]+'"'+' '+'since:'+date_1_+' '+'until:'+date_1 for i in query]
    print(query)
    return query

queries = []

for event in tqdm(df_events.iterrows()):
    doc_index = event[0]
    doc_date = event[1]['date']
    doc_class = event[1]['class']
    doc_title = event[1]['title']
    doc = event[1]['des_clean']
    class_text = doc_class.replace('and','')
    query = get_query(doc,class_text,doc_index,doc_date)
    queries.append(query)

query = db.current_event.find({},{'_id':1})
ids = []
for i in query:
    ids.append(i['_id'])

requests = [UpdateOne({'_id': _id}, {'$set': {'queries':queries[index]}}) for index,_id in tqdm(enumerate(ids))]
try:
    result = db.current_event.bulk_write(requests)
    pprint(result.bulk_api_result)
except BulkWriteError as bwe:
    pprint(bwe.details)
