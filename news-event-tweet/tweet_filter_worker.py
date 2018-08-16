import os
import re
import codecs
import pandas as pd
from tqdm import tqdm
import spacy
nlp = spacy.load('en_core_web_md')

import json    
from bson import json_util
import redis

r = redis.StrictRedis(host='localhost',port=6379, db=0)

import pymongo

client = pymongo.MongoClient('localhost:27017')
db = client.tweet


current_event_filter = {'_id':1,'event.date':1,'type':1,'event.title':1,'abstracts':1}

types = {
    0:'others',
    1:'army attack and conflict',
    2:'disaster and accident',
    3:'law and crime',
    4:'political and election',
    5:'international relation',
    6:'science and technology',
    7:'business and economic',
    8:'art and culture',
    9:'sports',
    10:'health and environment'
}


def flatten_event(e):
    return {'id':e['_id'],'category':types[e['type']],           'date':e['event']['date'],'title':e['event']['title'],           'description':e['abstracts'][0]['abstract'],           'media':e['abstracts'][0]['media'] if 'media' in e['abstracts'][0].keys() else [] }


query_dict = {'_id':1,'tweet.lang':1,'tweet.standard_text':1,              'tweet.created_at':1,'tweet.hashtags':1,'tweet.action.favorites':1,              'tweet.action.replies':1,'tweet.action.retweets':1,'tweet.user.data_name':1,              'tweet.user.screen_name':1,'tweet.user.userbadges':1,'tweet.action.is_retweet':1,              'tweet.is_reply':1,'tweet.mentions':1,'tweet.urls':1}


def flatten(i):
    return {'id':i['_id'],'lang':i['tweet']['lang'],            'text':re.sub('https?:\/\/\w+\.\w+\/\w+','',i['tweet']['standard_text']),            'created_at':str(i['tweet']['created_at']),            'hashtags':'#'.join(i['tweet']['hashtags']) if i['tweet']['hashtags'] else 0,            'favorites':i['tweet']['action']['favorites'],'replies':i['tweet']['action']['replies'],            'retweets':i['tweet']['action']['retweets'],'data_name':i['tweet']['user']['data_name'],            'screen_name':i['tweet']['user']['screen_name'],'userbadges':i['tweet']['user']['userbadges'],            'is_retweet':i['tweet']['action']['is_retweet'],'is_reply':i['tweet']['is_reply'],            'mentions':'@'.join(i['tweet']['mentions']) if i['tweet']['mentions'] else 0 ,            'has_url':len(i['tweet']['urls']) != 0}

def get_all_task_dataset(event):
    event_des_doc = nlp(event['description'])
    filter_dict = {'event_id':event['id']}
    records = [flatten(i) for i in tqdm(db.pos.find(filter_dict,query_dict))] + [flatten(i) for i in tqdm(db.paper.find(filter_dict,query_dict))]
    df_tweets = pd.DataFrame.from_records(records,index=range(len(records)),columns=['id', 'text','created_at','lang',                                                                                 'hashtags','mentions','has_url',                                                                                 'is_retweet','is_reply','retweets','replies',                                                                                'favorites','data_name','screen_name','userbadges'])
    df_tweets['boe_cosine'] = df_tweets['text'].apply(lambda x:event_des_doc.similarity(nlp(x)))
    df_tweets.drop_duplicates(['boe_cosine'],inplace=True)
    df_tweets.sort_values(by='boe_cosine',ascending=False,inplace=True)
    tweets = df_tweets[df_tweets['boe_cosine'] > 0.75][:60].sort_values(by='id').to_dict(orient='records')
    event['id'] = str(event['id'])
    return json.dumps({'event':event,'tweets':tweets})


def process_task(queue):
    event_id = json.loads(queue['event_id'],object_hook=json_util.object_hook)
    event = db.current_event.find_one({'_id':event_id},current_event_filter)
    return flatten_event(event)

            
if __name__ == '__main__':
    with codecs.open('demo.jsonl','a+',encoding='utf-8') as f:
        print('data_worker start!')
        while True:
            queue = r.lpop('task:data')
            if queue:
                print('data_worker process!')
                f.write(get_all_task_dataset(process_task(json.loads(queue)))+'\n')