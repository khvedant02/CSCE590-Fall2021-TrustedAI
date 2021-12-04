# Vedant Khandelwal - Subreddit Topic Model

from gensim.models.phrases import Phraser 
from gensim.models import Phrases 
import pickle as pkl 
import pandas as pd 
import numpy as np 
import nltk 
import re 
import string

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

df = pkl.load(open("Anxiety.pickl", 'rb'), encoding='latin-1')
sent = df['text'].values.tolist()
fil_sent = []
for i in sent:
    if(str(i)!='nan' and str(i)!='None' and str(i)!='none'):
        fil_sent.append(i)
pre_sent = preprocess(fil_sent)

from gensim.models import Word2Vec
model = Word2Vec(sentences=pre_sent, window=5, min_count=1, workers=16)

import os
arr = os.listdir('./')
sent = list()
for filename in arr:
    df = pkl.load(open(filename, 'rb'), encoding='latin-1')
    sub = df['text'].values.tolist()
    sub = [i for i in sub if str(i)!='nan' and str(i)!='none' and str(i)!='None']
    sent = sent+sub

pre_sent = preprocess(sent)
from gensim.models import Word2Vec
model = Word2Vec(sentences=pre_sent, window=5, min_count=1, workers=16)

MBigram, MTrigram = train_ngram(pre_sent)
ngram_corpus = extract_ngrams(MBigram, MTrigram, pre_sent)
ngram_corpus = [[i for i in doc if i.count("_")>0] for doc in ngram_corpus]
chunksize = 1000
passes = 20
iterations = 100
eval_every = None

dictionary = gensim.corpora.Dictionary(ngram_corpus)
corpus = [dictionary.doc2bow(doc) for doc in ngram_corpus]
temp = dictionary[0]
id2word = dictionary.id2token
model = LdaMulticore(
    corpus=corpus,
    id2word=id2word,
    chunksize=chunksize,
    iterations=iterations,
    num_topics=50,
    passes=passes)
pkl.dump(dictionary, open("Ngram_LDA_Dict_Addiction.pkl", 'wb'))
model.save("NGram_Lda_Addiction.model")