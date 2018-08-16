import json,
import requests
import xmltodict
"""
## nlp services
`
docker run -p "127.0.0.1:8080:80" jgontrum/spacyapi:en_v2
docker run -p 9000:9000 phiedulxp/lxp:corenlp java -mx6g -cp *  edu.stanford.nlp.pipeline.StanfordCoreNLPServer 9000
docker run -p 8000:8000 allennlp/allennlp [python -m allennlp.run serve]
docker run -ti -p 8000:8000 -v e:\docker\an:/tmp allennlp/allennlp python -m allennlp.run serve
docker run -ti -p 7000:12345 lixiepeng/lxp:ltp /ltp_server --last-stage all
`
"""

model_ref = {
    'ner':'named-entity-recognition',
    'cr':'coreference-resolution',
    'srl':'semantic-role-labeling',
    'te':'textual-entailment',
    'mc':'machine-comprehension',
}
class AllenNLP:
    '''
    Wrapper for allennlp Restful API
    'ner':'named-entity-recognition', # {'sentence':}->{'logits':,'mask':,'tags':,'words':}
    'cr':'coreference-resolution',    # {'document':,}->{ "antecedent_indices":, "clusters":, "document": "predicted_antecedents"}
    'srl':'semantic-role-labeling',   # {'sentence':,}-{'tokens':,'verbs':,'words':}
    'te':'textual-entailment',        # {'hypothesis':,'.premise':}-{'label_logits':,'label_probs':,}
    'mc':'machine-comprehension',     # {'passage':,'question':}->  {"best_span":,"best_span_str":,
                                                                          "passage_question_attention":,
                                                                          "passage_tokens":,"question_tokens":,
                                                                          "span_end_logits":,"span_end_probs":,
                                                                          "span_start_logits":,"span_start_probs":,}
    '''
    def __init__(self, host='127.0.0.1', port='8000'):
        self.host = host
        self.port = port
    
    def annotate(self, data, model_name):
        self.model = model_ref[model_name]
        self.request_url = 'http://'+self.host+':'+self.port+'/'+'predict/'+self.model
        self.data = data
        try:
            res = requests.post(url=self.request_url,
                                json=self.data,
                                headers={'Connection': 'close'})
            return res.json()
        except Exception as e:
            print(e)

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

class SematicNLP:
    '''
    A class for event sematic graph nlp service 
    '''
    def __init__(self,host='localhost',
                 allennlp_port='8000',
                 corenlp_port='9000',
                 spacy_model='en_core_web_md'):
        self.allennlp = AllenNLP(host=host,port=allennlp_port)
        self.corenlp = StanfordCoreNLP(host=host,port=corenlp_port)
        self.spacy = spacy.load(spacy_model)

    def ner(self,doc):
        corenlp_ner = {}
        for sentence in self.corenlp.annotate(doc,properties={'annotators':'ner'})['sentences']:
            for entity in sentence['entitymentions']:
                corenlp_ner[entity['text']] = entity['ner']
                
        spacy_ner = {}
        [spacy_ner.update({ent.text:ent.label_}) for ent in self.spacy(doc).ents]
        spacy_ner.update(corenlp_ner)
        return spacy_ner
        
    def openie(self):
        relation = []
        for sentence in self.corenlp.annotate(doc,properties={'annotators':'openie'})['sentences']:
            for rel in sentence['openie']:
                relation.extend(rel)
        return relation
    
    def coref(self,doc):
        doc = self.allennlp.annotate({'document':doc},'cr')
        clusters = []
        for cluster in doc['clusters']:
            temp_cluster = []
            for span in cluster:
                temp_cluster.append(' '.join(doc['document'][span[0]:span[1]+1]))
            clusters.append(temp_cluster)
        return clusters
        
    def kbp(self,doc):
        relation = []
        for sentence in self.corenlp.annotate(doc,properties={'annotators':'kbp'})['sentences']:
            for rel in sentence['kbp']:
                relation.extend(rel)
        return relation
    
    def srl(self,sentence):
        result = self.allennlp.annotate({'sentence':sentence},'srl')
        descriptions =[verb['description'] for verb in result['verbs']]  
        props = [dict(role_pattern.findall(description)) for description in descriptions]
        return props
    
    def __call__(self,doc):
        #corenlp_annotate =  self.corenlp.annotate(doc,properties={'annotators':'ner,openie,coref,kbp'})['sentences']:
        #allennlp_annotate = 
        pass
        
class Ltp:
    
    def __init__(self, host='127.0.0.1', port='7000'):
        self.host = host
        self.port = port

    def annotate(self, doc,x='xml',t='all'):
        self.server_url = 'http://'+self.host+':'+self.port+'/ltp'
        data = {
            's': doc,
            'x': 'n',
            't': 'all'}
        try:
            res = requests.post(self.server_url,
                                data=data, 
                                headers={'Connection': 'close'})
            return json.dumps(xmltodict.parse(res.content))
        except Exception as e:
            print(e)

'''
import spacy
from spacy import displacy
nlp = spacy.load('en_core_web_md')
from IPython.display import display, HTML
def display_doc(doc,style='ent'):
    display(HTML(displacy.render(nlp(doc), style=style)))
'''

if __name__ == '__main__':
    import spacy
    nlp = spacy.load('en_core_web_sm')
    ltp = Ltp()
    ltp.annotate(doc='我爱北京天安门。')
    example = '''A military helicopter surveying the damage, carrying the Governor of Oaxaca Alejandro Murat Hinojosa and Mexico's Secretary of the Interior Alfonso Navarrete Prida, crashes over Jamiltepec, killing 13 people on the ground. These deaths are the only known ones related to the earthquake reported so far. '''
    anlp = AllenNLP()
    out1 = anlp.annotate({'sentence':example},'ner')
    snlp = StanfordCoreNLP()
    demo1 = snlp.annotate(example,properties={'annotators':'kbp'}) ## kbp(tokenize,ssplit,pos,depparse,lemma,ner,coref,kbp)