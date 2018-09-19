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

NER_ = set(['person','misc','money','number','ordinal','percent','date','time','duration','set','email','url'])
phrase_dict = {line.split('\t')[1].strip():float(line.split('\t')[0]) \
               for line in codecs.open('./AutoPhrase.txt','r',encoding='utf-8').readlines()}

def standard_str(string):
    return string.replace(' ','_')\
                    .replace('/','_')\
                    .replace('-','_')\
                    .replace(':','_')

def annotate_sentences(text):
    out = snlp.annotate(text, properties={ 'annotators':'ner,depparse','outputFormat': 'json'})
    #import pdb;pdb.set_trace()
    triples = []
    for sentence in out['sentences']:
        #print(sentence['index'])
        triples.extend(get_sentence_triples(sentence))
    return triples
    
def get_ner_phrase(text):
    text = text.split('_')
    phrase = [text[-1]]
    for i in range(2,len(text)+1):
        if ' '.join(text[-i:]) in phrase_dict.keys() and phrase_dict[' '.join(text[-i:])]>0.1:
            phrase.append('_'.join(text[-i:]))
    return phrase

def get_com_phrase(tokens):
    text = text.split('_')
    phrase = [text[-1]]
    for i in range(2,len(text)+1):
        #print(' '.join(text[-i:]))
        if ' '.join(text[-i:]) in phrase_dict.keys() and phrase_dict[' '.join(text[-i:])]>0.1:
            phrase.append('_'.join(text[-i:]))
    return phrase

def dep_rules(v=None,dep=None):
    if dep['d'] in set(['parataxis','conj:or']):return (dep['g'],'/r/tweet/dep/para',dep['dd'])
    if dep['d'] in set(['nmod:including']):return (dep['dd'],'/r/tweet/dep/IsA',dep['g'])
    if dep['d'] in set(['nmod:in','nmod:on','nmod:into','nmod:at','nmod:near','nmod:under',\
                        'nmod:off','nmod:after','nmod:across','nmod:during','nmod:through',\
                        'nmod:following']):return (dep['g'],'/r/tweet/dep/'+dep['d'],dep['dd'])
    if dep['gp'].startswith('N') and dep['ddp'].startswith('N'):
        if dep['d'] in set(['nmod:by','nmod:from','nmod:due_to']):return (dep['dd'],'/r/tweet/dep/cause',dep['g'])
    if dep['gp'].startswith('V') and dep['ddp'].startswith('V'):
        if dep['d'] in set(['advcl:after','advcl:while','dep','acl',\
                            'ccomp','conj','xcomp','conj:and']):return (dep['dd'],'/r/tweet/dep/after',dep['g'])
    if dep['gp'].startswith('V') and dep['ddp'].startswith('N'):
        try:
            if dep['d'] in set(['dobj','nsubjpass','nmod:with','nmod:to','acl:to']):v[dep['g']]['t'].add(dep['dd'])
            if dep['d'] in set(['nsubj','nmod:agent','nsubj:xsubj']):v[dep['g']]['s'].add(dep['dd'])
        except:
            import pdb;pdb.set_trace()

def get_sentence_triples(sentence):
    # token
    tokens_dict = {}
    [tokens_dict.update({t['index']:{'token':[standard_str(t['lemma'].lower())],'pos':t['pos']}}) \
     for t in sentence['tokens'] ]
    # entity
    entity = [{'text':standard_str(e['text'].lower()),'ner':e['ner'].lower(),\
               'tokenBegin':e['tokenBegin']+1,'tokenEnd':e['tokenEnd'],} \
              for e in sentence['entitymentions']]
    dep = sentence['enhancedPlusPlusDependencies']
    headword = defaultdict(list)
    for d in dep:
        if d['dep'] == 'compound':
            headword[d['governor']].append(d['dependent'])   
    entity_triples = []
    for ent in entity:
        if ent['ner'] in NER_:
            if ent['tokenEnd'] in headword.keys():
                tokens_dict[ent['tokenEnd']].update({'token':[ent['ner']],'pos':'NER'})
                for t in headword[ent['tokenEnd']]:
                    tokens_dict.pop(t)
            else:
                tokens_dict[ent['tokenEnd']].update({'token':[ent['ner']],'pos':'NER'})
        else:
            if ent['tokenBegin'] == ent['tokenEnd']:
                tokens_dict[ent['tokenBegin']]['token'].append(ent['ner'])
                entity_triples.append(('/c/en/'+tokens_dict[ent['tokenBegin']]['token'][0],'/r/tweet/IsA','/c/en/'+ent['ner']))
                if ent['ner'] == 'cause_of_death':
                    entity_triples.append(('/c/en/'+tokens_dict[ent['tokenBegin']]['token'][0],'/r/tweet/Cause','/c/en/death'))
            else:
                ner_phrases = get_ner_phrase(ent['text'])
                for phrase in ner_phrases:
                    if ent['ner'] == 'cause_of_death':entity_triples.append(('/c/en/'+phrase,'/r/tweet/Cause','/c/en/'+'death'))
                    entity_triples.append(('/c/en/'+phrase,'/r/tweet/IsA','/c/en/'+ent['ner']))
                for i in range(len(ner_phrases)):
                    for j in range(i+1,len(ner_phrases)):
                        entity_triples.append(('/c/en/'+ner_phrases[j],'/r/tweet/IsA','/c/en/'+ner_phrases[i]))
                com_phrases = []
                if ent['tokenEnd'] in headword.keys():
                    tokens = [tokens_dict[t]['token'][0] for t in headword[ent['tokenEnd']]] + [tokens_dict[ent['tokenEnd']]['token'][0]]
                    com_phrases = get_ner_phrase('_'.join(tokens))
                    for t in headword[ent['tokenEnd']]:
                        tokens_dict.pop(t)
                phrases = set(ner_phrases+com_phrases)
                #print(phrases)
                tokens_dict[ent['tokenEnd']]['token'].append(ent['ner'])
                for phrase in phrases:
                    tokens_dict[ent['tokenEnd']]['token'].append(phrase)
                tokens_dict[ent['tokenEnd']]['token'] = list(set(tokens_dict[ent['tokenEnd']]['token']))
    # dep
    tokens_dict = {k:v for k,v in tokens_dict.items() if v['pos'].startswith('N') or v['pos'].startswith('V')}
    tokens_index_set = set(tokens_dict.keys())
    dep = [{'d':d['dep'],'g':d['governor'],'gp':tokens_dict[d['governor']]['pos'],\
        'dd':d['dependent'],'ddp':tokens_dict[d['dependent']]['pos']} \
        for d in dep if d['dep'] != 'compound' and\
       d['dependent'] in tokens_index_set and\
       d['governor'] in tokens_index_set] 
    V = set([d['g'] for d in dep if d['gp'].startswith('V')])
    V_fact = {v:{'s':set(),'t':set()} for v in V}   
    dep_triples_index = [dep_rules(v=V_fact,dep=d) for d in dep if dep_rules(v=V_fact,dep=d)]   
    dep_triples = []
    for dt in dep_triples_index:
        #print(dt)
        for s in tokens_dict[dt[0]]['token']:
            for t in tokens_dict[dt[2]]['token']:
                dep_triples.append(('/c/en/'+s,dt[1],'/c/en/'+t)) 
    # fact   
    fact_triples = []
    for k,v in V_fact.items():
        if len(v['s'])>0 and len(v['t'])>0:
            for s in v['s']:
                for t in v['t']:
                    for ss in tokens_dict[s]['token']:
                        for rr in tokens_dict[k]['token']:
                            for tt in tokens_dict[t]['token']: 
                                #print(ss,rr,tt)
                                fact_triples.append(('/c/en/'+ss,'/r/tweet/open/'+rr,'/c/en/'+tt))
    return entity_triples + dep_triples + fact_triples

def safe_load(text):
    try:
        return json.loads(text)
    except:
        return {}

def category_filter(sample):
    if not sample:
        return False
    return sample['event']['category'].startswith('disaster') or sample['event']['category'].startswith('army')

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
    event_triples = [{'head':e[0],'rel':e[1],'tail':e[2]} for e in event_triples]
    return event_triples

import dask.bag as db
import dask.dataframe as df

if __name__ == '__main__':
    files = db.read_text('./news_event.jsonl')\
    .map(safe_load)\
    .filter(category_filter)\
    .map(get_event_triples)\
    .flatten()\
    .to_dataframe()\
    .compute()\
    .to_csv('./tweet-triples.csv',index=False)
    print(files)

'''
v2: 
catgory:accdient&disaster,attack&conflict
triples:openIE,dep-rel[remove duplicated],keep several various/general/specific version of concept/relation
'''