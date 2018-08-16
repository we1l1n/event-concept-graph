import re
import random
from datetime import datetime, timedelta
from tqdm import tqdm

from bs4 import BeautifulSoup
import requests
headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.81 Safari/537.36'}

from pprint import pprint
import pymongo
from pymongo import InsertOne
from pymongo.errors import BulkWriteError
client = pymongo.MongoClient('localhost:27017')
db = client.tweet

Portal_url = 'https://en.wikipedia.org/wiki/Portal:Current_events/'
years = [2010+i for i in range(8)]
months = ["January","February","March","April","May","June","July","August","September","October","November","December"]
page_urls = [Portal_url+m+'_'+str(y) for y in years for m in months][6:-2]

def trans_date(date_str):
	month = date_str.split('_')[1]
	month_num = months.index(month)+1
	date_str = date_str.replace(month,'0'+str(month_num) if month_num<10 else str(month_num))
	date_str = date_str.replace('_','-')
	return date_str

def get_events_from_page(page_url):
	res = requests.get(page_url,headers=headers)
	soup = BeautifulSoup(res.text,'lxml')
	tables = soup.find_all('table',attrs={'class':'vevent'})
	events = []
	for t in tables:
		try:
			date = trans_date(t.find_previous_sibling().get('id'))
		except:
			date = (datetime.strptime(date,'%Y-%m-%d')+timedelta(days=1)).strftime('%Y-%m-%d')
		td = t.find('td',attrs={'class':'description'})
		if td.dl == None:
			types = [p.get_text() for p in td.find_all('p',recursive=False)]
		else:
			types = [dl.get_text() for dl in td.find_all('dl',recursive=False)]
		for type_,ul in zip(types,[ul for ul in td.find_all('ul',recursive=False)]):
			for li in ul.find_all('li',recursive=False):
				if li.ul == None:
					events.append({'date':date,
							   'class':type_.strip(),
							   'title':'',
							   'description':li.get_text().strip()})
				else:
					events.append({'date':date,
								   'class':type_.strip(),
								   'title':li.a.get_text(),
								   'description':li.ul.get_text().strip()})
	return events
	
if __name__ == '__main__':
	event_dicts = []
	for page_url in tqdm(page_urls):
		event_dicts.extend(get_events_from_page(page_url))
	requests_ = [InsertOne({'_id': hash(i['date']+i['title']+str(random.randint(0,100))),'event':i}) for i in tqdm(event_dicts)]
	try:
		result = db.current_event.bulk_write(requests_)
		pprint(result.bulk_api_result)
	except BulkWriteError as bwe:
		pprint(bwe.details)
	client.close()