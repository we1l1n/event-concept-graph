import json
import requests
class StanfordCoreNLP:
    '''
    Wrapper for Starford Corenlp Restful API
    annotators:"truecase,tokenize,ssplit,pos,lemma,ner,regexner,parse,depparse,openie,coref,kbp,sentiment"
    nlp = StanfordCoreNLP()
    output = nlp.annotate(text, properties={ 'annotators':'kbp','outputFormat': 'json',})
    output.keys()
    dict_keys(['sentences', 'corefs'])
    
    '''

    def __init__(self, host='127.0.0.1', port='9000'):
        self.host = host
        self.port = port

    def annotate(self, data, properties=None, lang='en'):
        self.server_url = 'http://'+self.host+':'+self.port
        properties['outputFormat'] = 'json'
        try:
            res = requests.post(self.server_url,
                                params={'properties': str(properties),
                                        'pipelineLanguage':lang},
                                data=data, 
                                headers={'Connection': 'close'})
            return res.json()
        except Exception as e:
            print(e)
        
snlp = StanfordCoreNLP(host='10.0.1.7')

from gensim.models import KeyedVectors
from gensim.similarities.index import AnnoyIndexer
wv = KeyedVectors.load_word2vec_format('glove.6B.300d.bin', binary=True)
annoy_index = AnnoyIndexer()
annoy_index.load('glove.6B.300d.index')
annoy_index.model = wv
vocab_disasters_and_accdients = dict(wv.most_similar(positive=['disaster','accident',], topn=25000, indexer=annoy_index))

from collections import defaultdict
from nltk.corpus import stopwords
list_stopWords=list(set(stopwords.words('english')))

NER_ = ['person','ordinal','number','date']
DEP_ = ['amod','punct','case','det','ROOT','dep']

def dep_to_str(tokens_dict,entity_dict,index):
    return entity_dict[index] if index in entity_dict.keys() else tokens_dict[index]['lemma']

def annotate_sentences(text):
    out = snlp.annotate(text, properties={ 'annotators':'ner,depparse','outputFormat': 'json'})
    #import pdb;pdb.set_trace()
    triples = []
    for sentence in out['sentences']:
        #print(sentence['index'])
        triples.extend(get_triples(sentence))
    return triples
    
def get_triples(sentence):
    # tokens 
    tokens_dict = {}
    [tokens_dict.update({t['index']:{'lemma':t['lemma'].lower(),'word':t['word'],'pos':t['pos']}}) for t in sentence['tokens']]
    # entity 
    entity = [{'text':e['text'].lower(),'ner':e['ner'].lower(),'tokenBegin':e['tokenBegin']+1,'tokenEnd':e['tokenEnd'],}\
              for e in sentence['entitymentions']]
    entity_dict = {}
    span_entity = [] # combine in dep
    entity_triples = []
    for e in entity:
        entity_dict.update({e['tokenEnd']:e['ner'] if e['ner'] in NER_ else e['text']})#[e['ner'],e['text']]})
        if e['ner'] not in NER_:
            entity_triples.append(('/c/en/'+e['text'],'/r/tweet/IsA','/c/en/'+e['ner']))
            #print('/c/en/'+e['text'],'/r/tweet/IsA','/c/en/'+e['ner'])
        if e['tokenEnd'] != e['tokenBegin']:
            span_entity.append((e['tokenBegin'],e['tokenEnd']))
    # dep triples
    # vocab_disasters_and_accdients
    dep = sentence['enhancedPlusPlusDependencies']
    dep = [{'from':d['governor'],'dep':d['dep'],'to':d['dependent']} for d in dep if d['dep'] not in DEP_ \
       and (d['dependent'],d['governor']) not in span_entity and d['dependentGloss'] not in list_stopWords\
      and d['governorGloss'] not in list_stopWords]# and d['dependentGloss'] in vocab_disasters_and_accdients.keys()\
      #and d['governorGloss'] in vocab_disasters_and_accdients.keys()]
    dep_triples =[('/c/en/'+dep_to_str(tokens_dict,entity_dict,d['to']),'/r/tweet/dep/'+d['dep'],\
                   '/c/en/'+dep_to_str(tokens_dict,entity_dict,d['from'])) for d in dep]
    # fact_triples filter from dep
    fact_dict = defaultdict(dict)
    for d in dep:
        if d['dep'].startswith('nsubj') or d['dep'].startswith('dobj'):
                fact_dict[d['from']].update({d['dep']:d['to']})
    fact_triples = []
    for k,v in fact_dict.items():
        if 'nsubj' in v.keys() and 'dobj' in v.keys():
            fact_triples.append(('/c/en/'+dep_to_str(tokens_dict,entity_dict,v['nsubj']),\
                                 '/r/tweet/open/'+dep_to_str(tokens_dict,entity_dict,k),\
                                 '/c/en/'+dep_to_str(tokens_dict,entity_dict,v['dobj'])))
    # cat all triples
    triples = entity_triples + dep_triples + fact_triples
    return triples

def safe_load(text):
    try:
        return json.loads(text)
    except:
        return {}

def category_filter(sample):
    if not sample:
        return False
    return sample['event']['category'].startswith('disaster')

def get_event_triples(sample):
    event_triples = []
    try:
        print(sample['event']['description'])
        event_triples.extend(annotate_sentences(sample['event']['description']))
    except:
        pass
    for index,tweet in enumerate(sample['tweets']):
        print(index)
        try:
            print(tweet['text'])
            event_triples.extend(annotate_sentences(tweet['text']))
        except:
            pass
    return event_triples


if __name__ == '__main__':
    import dask.bag as db
    news_event = db.read_text('./news_event.jsonl').map(json.loads)
    news_event_disasters_and_accdients = news_event.filter(category_filter)
    #sample = news_event_disasters_and_accdients.take(1)[0]
    event_triples = news_event_disasters_and_accdients.map(get_event_triples)
    triples = []
    for e in event_triples:
        triples.extend(e)
    import dask.dataframe as df
    df_triples =  df.from_array(triples,columns=['head','rel','tail'])
    df_triples.to_csv('./triples-*.csv')  