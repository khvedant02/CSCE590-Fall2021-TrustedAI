#Vedant Khandelwal - All sub-reddit laguage models

from gensim.models.phrases import Phraser 
from gensim.models import Phrases 
import pickle as pkl 
import pandas as pd 
import numpy as np 
import nltk 
import re 
from gensim.models import Word2Vec
import string
from gensim.parsing.preprocessing import stem_text
import gensim
import os

nltk.download('stopwords') 
nltk.download('wordnet') 
nltk.download('punkt') 

def black_txt(token, stop_words_, common_terms):  
    return  token not in stop_words_ and token not in list(string.punctuation)  and len(token)>2  
 
def clean_txt(text, stop_words_, common_terms):  
    clean_text = []  
    clean_text2 = []  
    #print(text)
    text = re.sub(r'http\S+', '',text)  
    #text = stem_text(text)  
    clean_text = [ word for word in nltk.word_tokenize(text.lower()) if black_txt(word, stop_words_, common_terms)]  
    clean_text2 = [word for word in clean_text if black_txt(word, stop_words_, common_terms)]  
    return " ".join(clean_text2) 
 
def preprocess(corpus):  
    stop = open('/work/vedant/Others/terrier-stop.txt','r')  
    stopString = stop.read()  
    common_terms = stopString.split()  
    stop_words_ = set(nltk.corpus.stopwords.words('english'))  
    #wn = WordNetLemmatizer()  
    temp_corpus = [nltk.word_tokenize(clean_txt(doc, stop_words_, common_terms)) for doc in corpus]  
    return temp_corpus  

def train_ngram(corpus): 
    bigram = Phrases(corpus, min_count=1, threshold=0.75) 
    bcorpus = bigram[corpus] 
    Model_bg = Phraser(bigram) 
    trigram = Phrases(bcorpus, min_count = 1, threshold = 0.5) 
    Model_tg = Phraser(trigram) 
    return Model_bg, Model_tg 

def extract_ngrams(BModel, TModel, corpus): 
    ngram = [TModel[BModel[doc]] for doc in corpus] 
    return ngram

import os
fpath = '/work/vedant/Others/subreddit/'
filelist = list()
for cat in ['depression/','anxiety/', 'addiction/']:
    arr = os.listdir(fpath+cat)
    arr = [fpath+cat+i for i in arr if i.count(".pickl")>0]
    for file in arr:
        filelist.append(file)

processed_sent = []
for file in filelist:
    df = pkl.load(open(file, 'rb'), encoding='latin-1')
    text = df['text'].values.tolist()
    text = [i.lower() for i in text if str(i)!='nan' and str(i)!='none' and str(i)!='None']  
    text = preprocess(text)
    processed_sent = processed_sent + text

MB, TB = train_ngram(text)
text = extract_ngrams(MB, TB, text)
model = Word2Vec(sentences = text, size = 300, window=5, min_count=1, workers=32)
print('Done')
model.save("/work/vedant/subreddit_word2vec.model")