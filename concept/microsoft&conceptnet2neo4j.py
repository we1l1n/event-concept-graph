import dask.dataframe as dd
df = dd.read_csv('data-concept-instance-relations.txt',sep='\t',names=['concept','instance','probability'])
#33377320
df['start'] = df['instance'].apply(lambda x:'/c/en/'+str(x).replace(',','').replace(' ','_'))
df['rel'] = '/r/microsoft/IsA'
df['end'] = df['concept'].apply(lambda x:'/c/en/'+str(x).replace(',','').replace(' ','_'))
df['weight'] = df['probability'].apply(lambda x:x/10000)
df.to_csv('neo/microsoft-rel-*.csv',encoding='utf-8',sep=',',header=False,columns=['start','rel','end','weight'],index=False)
df['start-label'] = 'CONCEPT'
df['end-label'] = 'CONCEPT'
df.to_csv('neo/microsoft-start-*.csv',encoding='utf-8',sep=',',header=False,columns=['start','start-label'],index=False)
df.to_csv('neo/microsoft-end-*.csv',encoding='utf-8',sep=',',header=False,columns=['end','end-label'],index=False)

import re
weight_re = re.compile('\"weight\":\s(.+)}')
#32755361
df = dd.read_csv('conceptnet-assertions-5.6.0.csv',sep='\t',names=['URI','rel','start','end','info'])
df['weight'] = df['info'].apply(lambda x:float(weight_re.search(x).groups()[0]) if x!= 'foo' else 0.0)
df.to_csv('neo/conceptnet-rel-*.csv',encoding='utf-8',sep=',',header=False,columns=['start','rel','end','weight'],index=False)
df['start-label'] = 'CONCEPT'
df.to_csv('neo/conceptnet-start-*.csv',encoding='utf-8',sep=',',header=False,columns=['start','start-label'],index=False)
df['end-label'] = df['end'].apply(lambda x:'URL' if x.startswith('http://') else 'CONCEPT')
df.to_csv('neo/conceptnet-end-*.csv',encoding='utf-8',sep=',',header=False,columns=['end','end-label'],index=False)

###########################################################################
#EN-ZH-noURL
import dask.dataframe as dd
df = dd.read_csv('data-concept-instance-relations.txt',sep='\t',names=['concept','instance','probability'])
#33377320
df['start'] = df['instance'].apply(lambda x:'/c/en/'+str(x).replace(',','').replace(' ','_'))
df['rel'] = '/r/microsoft/IsA'
df['end'] = df['concept'].apply(lambda x:'/c/en/'+str(x).replace(',','').replace(' ','_'))
df['weight'] = df['probability'].apply(lambda x:x/10000)
df.to_csv('neo-en-zh-noURL/microsoft-rel-*.csv',encoding='utf-8',sep=',',header=False,columns=['start','rel','end','weight'],index=False)
node = df['start'].append(df['end']).drop_duplicates()
node.to_csv('neo-en-zh-noURL/microsoft-node-*.csv',encoding='utf-8',sep=',',header=False,index=False)
from uri import *
import re
weight_re = re.compile('\"weight\":\s(.+)}')
#32755361
df = dd.read_csv('conceptnet-assertions-5.6.0.csv',sep='\t',names=['URI','rel','start','end','info'])
lang = ['en','zh']
df['filter'] = df['start'].apply(lambda x:get_uri_language(x) in lang) & df['end'].apply(lambda x:not is_absolute_url(x) and get_uri_language(x) in lang)
#len(df[df['filter']])
#4143479
df = df[df['filter']]
df['weight'] = df['info'].apply(lambda x:float(weight_re.search(x).groups()[0]) if x!= 'foo' else 0.0)
df.to_csv('neo-en-zh-noURL/conceptnet-rel-*.csv',encoding='utf-8',sep=',',header=False,columns=['start','rel','end','weight'],index=False)
node = df['start'].append(df['end']).drop_duplicates()
node.to_csv('neo-en-zh-noURL/conceptnet-node-*.csv',encoding='utf-8',sep=',',header=False,index=False)