import codecs
import json
import requests
import multiprocessing as mp
from tqdm import tqdm

model_ref = {
    'ner':'/named-entity-recognition',
    'cr':'/coreference-resolution',
    'srl':'/semantic-role-labeling',
    'te':'/textual-entailment',
    'mc':'/machine-comprehension',
    '':''
}
class AllenNLP:
    def __init__(self, host='127.0.0.1', port='8000'):
        self.host = host
        self.port = port
    
    def annotate(self, data, model_name):
        self.model = model_ref[model_name]
        self.request_url = 'http://'+self.host+':'+self.port+'/'+'predict'+self.model
        self.data = data
        try:
            res = requests.post(url=self.request_url,
                                json=self.data,
                                headers={'Connection': 'close'})
            return res.json()
        except Exception as e:
            print(e)
            

def get_top3_cat(text):
    anlp = AllenNLP()
    res = anlp.annotate({'event_tweet':text},'')
    try:
        return dict(sorted(res.items(),key=lambda x:x[1],reverse=True)[:3])
    except:
        return {}

def update_top3_cat(d):
    d.update({'top3_cat':get_top3_cat(d.get('description') if d.get('description',None) else d.get('text'))})

def worker(q,line):
    print(line['event']['date'])
    print(len(line['tweets']))
    print(q.qsize())
    if len(line['tweets']) >= 10:
      update_top3_cat(line['event'])
      for tweet in line['tweets']:
          update_top3_cat(tweet)
      q.put(line)
    
if __name__ == '__main__':
    update_f = codecs.open('cat_news_event.jsonl','a+',encoding='utf-8')
    f = codecs.open('news_event.jsonl','r',encoding='utf-8')
    manager = mp.Manager()
    q = manager.Queue()
    #q.full()
    #q.empty()
    #q.qsize()
    lines = []
    for line in f.readlines():
        try:
            lines.append(json.loads(line))
        except:
            pass
    update_lines = []
    pool = mp.Pool(16)
    res = [pool.apply_async(worker,args=(q,line,)) for line in tqdm(lines)]
    pool.close()
    pool.join()
    #import pdb
    #pdb.set_trace()
    while q.qsize() != 0:
       update_lines.append(q.get()) 
    update_lines = [json.dumps(i)+'\n' for i in update_lines]
    update_f.writelines(update_lines)
    update_f.close()