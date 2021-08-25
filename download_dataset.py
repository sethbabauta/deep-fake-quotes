from bs4 import BeautifulSoup as bs
import requests, json
import csv

import os
from dotenv import load_dotenv
load_dotenv()
tags = eval(os.getenv("tags_vocab"))
urls = ["https://www.goodreads.com/quotes/tag/" + tag for tag in tags]# all of the tags we want to search (obviously)

def get_text(elem):
	text = str(elem).split("\n")[1].strip("  ")
	text = text.replace("<br/>", " ")
	return text

def collect_data(url, pages=5):
	data = []
	for i in range(pages):
		res = requests.get(url+f"?page={i}")
		content = bs(res.content, "html.parser")
		elems = content.findAll(attrs={"class": "quoteText"})
		for elem in elems:
			t = get_text(elem)
			data.append(t)

	return data

def store_data(data, path="data.csv"):
	data = [[x] for x in data]
	with open(path, "w") as f:
		writer = csv.writer(f)
		writer.writerows(data)
		return True

	return False

data = []
for u in urls:
	collected = collect_data(u)
	data = data + collected

store_data(data)

print(f"Collected {len(data)} samples")
print("Stored data.")