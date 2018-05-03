import hug
import py2neo
import uri

from py2neo import Graph
graph = Graph()

@hug.get('/conceptualization')
@hug.post('/conceptualization')
def conceptualization(words,lang='en',rel='/microsoft/IsA',topk=10,su='p'):
    '''
    requests.post('http://127.0.0.1:8000/conceptualization',data=data).json()
    data = {'su': 'b', 'topk': 1, 'words': 'apple'}
    [{'Concept': '/c/en/apple', 'Instance': '/c/en/gala', 'is a(n)': 0.0250000004}]
    data = {'su': 'p', 'topk': 1, 'words': 'apple'}
    [{'Concept': '/c/en/fruit','Instance': '/c/en/apple','is a(n)': 0.6315000057}]
    data = {'rel': '/IsA', 'su': 'b', 'words': 'color', 'topk':1}
    [{'Concept': '/c/en/color', 'Instance': '/c/en/blue', 'is a(n)': 6.9279999733}]
    data = {'rel': '/IsA', 'su': 'p', 'words': 'color', 'topk':1}
    [{'Concept': '/c/en/cause_by_reflection_of_light','Instance': '/c/en/color','is a(n)': 1.0}]
    data = {'su': 'p', 'words': ['apple','pie'], 'topk':1}
    [{'Concept': '/c/en/food', 'prob': 0.0029491198}]
    data = {'su': 'b', 'words': ['apple','pie'], 'topk':1}
    [{'Instance': '/c/en/bread', 'freq': 115}]
    '''
    words = locals()['words']
    lang = locals()['lang']
    rel = locals()['rel']
    topk = int(locals()['topk'])
    su = locals()['su']
    num_words = 1 if isinstance(words,str) else len(words)
    if num_words == 1:
        word = words
        text = uri.concept_uri(lang,word)
        r = uri.join_uri('/r',rel)
        query = 'MATCH (i:Concept)-[r:`%s`]->(c:Concept {conceptId:"%s"}) RETURN i.conceptId as Instance, r.weight as `is a(n)`,c.conceptId as Concept ORDER BY r.weight DESC LIMIT %d;'%(r,text,topk) if su == 'b' else 'MATCH (i:Concept {conceptId:"%s"})-[r:`%s`]->(c:Concept) RETURN i.conceptId as Instance, r.weight as `is a(n)`,c.conceptId as Concept ORDER BY r.weight DESC LIMIT %d;'%(text,r,topk)
        print(locals())
        return graph.run(query).data()
    elif num_words == 2:
        texts = [uri.concept_uri(lang,word) for word in words]
        r = uri.join_uri('/r',rel)
        query = 'MATCH (a:Concept {conceptId:"%s"})-[r1:`%s`]->(c:Concept)<-[r2:`%s`]-(b:Concept {conceptId:"%s"}) USING INDEX a:Concept(conceptId) USING INDEX b:Concept(conceptId) MATCH (c:Concept)<-[r:`%s`]-(o:Concept) WHERE o <> a and o <> b WITH o, count(*) AS freq ORDER BY freq DESC LIMIT %d RETURN o.conceptId AS Instance, freq;'%(texts[0],r,r,texts[1],r,topk) if su == 'b' else 'MATCH (a:Concept {conceptId:"%s"})-[r1:`%s`]->(c:Concept)<-[r2:`%s`]-(b:Concept {conceptId:"%s"}) USING INDEX a:Concept(conceptId) USING INDEX b:Concept(conceptId) RETURN c.conceptId AS Concept, r1.weight*r2.weight AS prob ORDER BY prob DESC LIMIT %d;'%(texts[0],r,r,texts[1],topk)
        print(locals())
        return graph.run(query).data()
    else:
        print(locals())
        return {'message':'null'}