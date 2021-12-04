# Vedant Khandelwal - training topic models for the category specific tweets

from sklearn.metrics.pairwise import cosine_similarity as cosim 
from gensim.models.phrases import Phraser 
from gensim.models import Phrases 
import pickle as pkl 
import pandas as pd 
import numpy as np 
import nltk 
import re 
from gensim.models import LdaModel 
from gensim.models import Word2Vec
import string
from gensim.parsing.preprocessing import stem_text
from gensim.models.ldamulticore import LdaMulticore
import gensim
from gensim.models.coherencemodel import CoherenceModel
 
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
    stop = open('/work/vedant/terrier-stop.txt','r')  
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
arr = os.listdir('./')
arr = [i for i in arr if i!="w2vmodel"]
sent = list()
for filename in arr:
    df = pkl.load(open(filename, 'rb'), encoding='latin-1')
    sub = df['text'].values.tolist()
    sub = [i for i in sub if str(i)!='nan' and str(i)!='none' and str(i)!='None']
    sent = sent+sub

chunksize = 500
iterations = 100
passes = 200
#cat category
cat = ['Depression', 'Anxiety', 'Addiction']
pre_sent = preprocess(sent)
bt, mt = train_ngram(pre_sent)
n_sent = extract_ngrams(bt, mt, pre_sent)
dictionary = gensim.corpora.Dictionary(n_sent)
corpus = [dictionary.doc2bow(doc) for doc in n_sent]
temp = dictionary[0]
id2word = dictionary.id2token
model = LdaMulticore(
    corpus=corpus,
    id2word=id2word,
    chunksize=chunksize,
    iterations=iterations,
    num_topics=55,
    passes=passes)
pkl.dump(dictionary, open("LDA_Dict_"+cat+".pkl", 'wb'))
model.save("Lda_"+cat+".model")


chunksize = 40000
passes = 20
iterations = 100
eval_every = None
dictionary = gensim.corpora.Dictionary(pre_sent)
corpus = [dictionary.doc2bow(doc) for doc in pre_sent]
temp = dictionary[0]
id2word = dictionary.id2token
model = LdaMulticore(
    corpus=corpus,
    id2word=id2word,
    chunksize=chunksize,
    iterations=iterations,
    num_topics=65,
    passes=passes)

pkl.dump(dictionary, open("LDA_Dict_Depression.pkl", 'wb'))
model.save("LdaNgram_Depression.model")