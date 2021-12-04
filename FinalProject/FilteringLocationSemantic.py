#Vedant Khandelwal - Dataset Filtering (Location) AND Semantic Filtering

import pandas as pd
import numpy as np
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
import pickle as pkl 
import json
from json import JSONDecodeError
import requests
from bs4 import BeautifulSoup
import bz2
from nltk import ngrams

def intersection(lst1, lst2): 
    return list(set(lst1) & set(lst2))

def lexicon_dict(csvpath):
    df = pd.read_csv(csvpath+"lex.csv")
    lexdict = dict()
    for cat in df.columns:
        temp = df[cat].values.tolist()
        temp = [i for i in temp if str(i)!='nan']
        check = []
        for i in temp:
            check.append(i.replace(" ","_"))
            if(i.count(" ")>1):
                tem = i.split(" ")
                tem = [i for i in tem if len(i)>2]
                check.append("_".join(tem))
                for j in tem:
                    check.append(j)
                if(len(tem)>1):
                    tem = ngrams(tem,2)
                    for k in tem:
                        check.append(k[0]+"_"+k[1])
            else:
                tem = i.split(" ")
                tem = [i for i in tem if len(i)>2]
                for j in tem:
                    check.append(j)
        check = [i.lower() for i in check]
        check = list(set(check))
        lexdict[cat] = check
    
    return lexdict

def allcountrylist():
    URL = 'https://en.wikipedia.org/wiki/List_of_alternative_country_names'
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, 'html.parser')
    data = list()
    table = soup.findAll('table', attrs={'class':'wikitable'})
    for i in table:
        table_body = i.find('tbody')
        rows = table_body.find_all('tr')  
        for row in rows:
            cols = row.find_all('td')
            cols = [ele.text.strip() for ele in cols]
            data.append([ele for ele in cols if ele])
    country = {}
    for i in data:
        if(len(i)>1):
            temp = i
            top = set()
            try:
                for j in temp[2].split(","):
                    top.add(j.split(" (")[0])
                country[temp[1].split(" (")[0].strip()] = ", ".join([i.rstrip() for i in top if i.count(")")==0 and i.count("]")==0])
                country[temp[1].split(" (")[0].strip()] = country[temp[1].split(" (")[0].strip()][3:]
            except IndexError:
                for j in temp[1].split(","):
                    top.add(j.split(" (")[0])
                country[temp[0].split(" (")[0].strip()] = ", ".join([i.rstrip() for i in top if i.count(")")==0 and i.count("]")==0])
                country[temp[0].split(" (")[0].strip()] = country[temp[0].split(" (")[0].strip()][3:]
    
    return country

def famouscountry(csvpath, country):
    df = pd.read_csv(csvpath+"famouscountry.csv")
    names = df.CountryName
    names = [i.strip() for i in names if i.strip()!='United States' and i.strip()!='China']
    allnameslist = []
    for name in names:
        allnameslist.append(name)
        for slang in country[name].split(", "):
            if(len(slang)>1):
                allnameslist.append(slang)
    allnameslist = list(set(allnameslist))

    return allnameslist

def locationfilter(allnameslist,tweet):
    #filtered = [tweet for tweet in tweets if len(intersection(allnameslist, nltk.word_tokenize(tweet)))>0]
    if(len(intersection(allnameslist, [i.lower() for i in nltk.word_tokenize(tweet)]))==0):
    #return filtered
        return 1
    else:
        return 0

def categoryfilter(categorylexicon, tweets):
    #filtered = [tweet for tweet in tweets if len(intersection(categorylexicon, nltk.word_tokenize(tweet)))>0]
    #print(categorylexicon, tweets)
    #raise KeyboardInterrupt
    if(len(intersection(categorylexicon, [i.lower() for i in nltk.word_tokenize(tweets)]))>0):
    #return filtered
        return 1
    else:
        return 0

def languagefilter(tweetobject):
    if(tweetobject['lang']=='en'):
        return 1
    else:
        return 0


csvpath = "/work/vedant/CSVDatasets/"
lexdict = lexicon_dict(csvpath)
for cat in lexdict:
    lexdict[cat] = [i.lower() for i in lexdict[cat]]
country = allcountrylist()
allnameslist = famouscountry(csvpath, country)
allnameslist = [i.lower() for i in allnameslist]

import os
arr = os.listdir()
arr = [i for i in arr if i.count(".json.bz2")>0]

cat_dict = {'Depression':[],'Addiction':[], 'Anxiety':[]}
for i in arr:
    with bz2.open(i, "rt") as bzinput:
        for line in enumerate(bzinput):
            try:
                tweet = json.loads(line[1])
            except JSONDecodeError:
                pass
            if(languagefilter(tweet)):
                try:
                    text = tweet['retweetedStatus']['text']
                    if(locationfilter(allnameslist, text)):
                        #parts.append(tweet['retweetedStatus']['text'])
                        for cat in ['Depression', 'Addiction', 'Anxiety']:
                            if(categoryfilter(lexdict[cat],text) and len(cat_dict[cat])<400001):
                                cat_dict[cat].append(text)
                            elif(len(cat_dict[cat])==400000):
                                continue
                except TypeError:
                    text = tweet['text']
                    if(locationfilter(allnameslist,text)):
                        #parts.append(tweet['text'])
                        for cat in ['Depression', 'Addiction', 'Anxiety']:
                            if(categoryfilter(lexdict[cat],text) and len(cat_dict[cat])<400001):
                                cat_dict[cat].append(text)
                            elif(len(cat_dict[cat])==400000):
                                continue

catdict = {}
for cat in lexdict:
    catdict[cat] = categoryfilter(lexdict[cat],parts)

pkl.dump(catdict, open("/work/vedant/COV_ResultsCode/6_2_cat.pkl","wb"))

for cat in cat_dict:
    print(cat, len(cat_dict[cat]))
import pickle as pkl
pkl.dump(cat_dict, open("Saved_Again.pkl", 'wb'))