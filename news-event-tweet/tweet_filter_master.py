from tqdm import tqdm
import pymongo
import redis
import json    
from bson import json_util

client = pymongo.MongoClient('localhost:27017')
db = client.tweet

r = redis.StrictRedis(host='localhost', port=6379, db=0)
r.delete('task:data')
events = [e for e in db.current_event.find({},{'_id':1})]

def send_message(eid):
	message = {'event_id':json.dumps(eid,default=json_util.default)}
	r.rpush('task:data',json.dumps(message))

if __name__ == '__main__':
	[send_message(e['_id']) for e in tqdm(events)]