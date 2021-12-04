#Vedant Khandelwal - Training Sub-Reddit Language models

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
    clean_text = [ word for word in nltk.word_tokenize(text.lower()) if black_txt(word, stop_words_, common_terms) and word.count(".")<1]  
    clean_text2 = [word for word in clean_text if black_txt(word, stop_words_, common_terms) and word.count("'")<1]   
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

arr = os.listdir("/work/vedant/Others/subreddit/depression/")
arr = ["/work/vedant/Others/subreddit/depression/"+i for i in arr if i.count(".pickl")>0]
arr = os.listdir("/work/vedant/Others/subreddit/addiction/")
arr = ["/work/vedant/Others/subreddit/addiction/"+i for i in arr if i.count(".pickl")>0]
arr = os.listdir("/work/vedant/Others/subreddit/anxiety/")
arr = ["/work/vedant/Others/subreddit/anxiety/"+i for i in arr if i.count(".pickl")>0]
files = arr+ arr2 + arr3

fil_sent = list()
for f in arr:
    df = pkl.load(open(f, "rb"), encoding='latin-1')
    sen = [s for s in df['text'].values.tolist() if str(s)!='nan' and str(s)!='None' and str(s)!='none']
    fil_sent = fil_sent + preprocess(sen)

MB, MT = train_ngram(fil_sent)
ngrams = extract_ngrams(MB, MT, fil_sent)

chunksize = 2000
passes = 20
iterations = 100
eval_every = None
dictionary = gensim.corpora.Dictionary(ngrams)
corpus = [dictionary.doc2bow(doc) for doc in ngrams]
temp = dictionary[0]
id2word = dictionary.id2token
model = LdaMulticore(
    corpus=corpus,
    id2word=id2word,
    chunksize=chunksize,
    iterations=iterations,
    num_topics=90,
    workers = 20,
    passes=passes)
pkl.dump(dictionary, open( "/work/vedant/COV_FinalExp/Ngram_LDA_Dict_Anxiety.pkl", 'wb'))
model.save("/work/vedant/COV_FinalExp/NGram_Lda_Anxiety.model")

dictionary = gensim.corpora.Dictionary(fil_sent)
corpus = [dictionary.doc2bow(doc) for doc in fil_sent]
temp = dictionary[0]
id2word = dictionary.id2token
model = LdaMulticore(
    corpus=corpus,
    id2word=id2word,
    chunksize=chunksize,
    iterations=iterations,
    num_topics=90,
    workers = 20,
    passes=passes)
pkl.dump(dictionary, open("/work/vedant/COV_FinalExp/LDA_Dict_Anxiety.pkl", 'wb'))
model.save("/work/vedant/COV_FinalExp/Lda_Anxiety.model")