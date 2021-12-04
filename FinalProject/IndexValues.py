# Vedant Khandelwal - Addiction Index

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


from gensim.models import Word2Vec

model = Word2Vec.load("/work/vedant/subreddit__300_word2vec.model")

df = pkl.load(open("/work/vedant/COV_FinalExp/alltweets.pkl","rb")) 
addict_tweets = df["Addiction"]
paddict_tweets = preprocess(addict_tweets)

df = pd.read_csv("/work/vedant/COV_FinalExp/lexicons.csv")
addict = [i.lower() for i in df['Addiction'].values.tolist() if str(i)!='nan']

ngram_addict_dict = pkl.load(open("/work/vedant/COV_FinalExp/Ngram_LDA_Dict_Addiction.pkl", "rb"))
ngram_addict_model = LdaModel.load("/work/vedant/COV_FinalExp/NGram_Lda_Addiction.model")

addict_dict = pkl.load(open("/work/vedant/COV_FinalExp/LDA_Dict_Addiction.pkl", "rb"))
addict_model = LdaModel.load("/work/vedant/COV_FinalExp/Lda_Addiction.model")

def topic_dictionary(num_tops, model):
    tdict = dict()
    for i in range(num_tops):
        temp = model.print_topic(i,10)
        temp = temp.split('" + ')
        temp = [i.split('*"')[1].replace('"',"") for i in temp]
        tdict[i] = temp
    
    return tdict

addict_tdict = topic_dictionary(50, addict_model)
ngram_addict_tdict = topic_dictionary(50, ngram_addict_model)

import numpy as np
def embedding(model, phrase):
    phrase = phrase.replace("*", "")
    sub = np.zeros(shape=(300,))
    try:
        sub = sub + model.wv[phrase]
        return sub
    except KeyError:
        if(phrase.count("_")>0):
            for word in phrase.split("_"):
                try:
                    sub = sub + model.wv[word]
                except KeyError:
                    continue
        elif(phrase.count(" ")>0):
            for word in phrase.split(" "):
                try:
                    sub = sub + model.wv[word]
                except KeyError:
                    continue
        return sub

def intersection(lst1, lst2): 
    if(len(lst1)==0 or len(lst2)==0):
        return 0
    else:
        cosim_matrix = cosim(lst1, lst2)
    filtered = [[idx for idx, val in enumerate(row) if val>=0.5 ]for row in cosim_matrix]
    filtered = [len(i) for i in filtered]
    filtered = sum(filtered)
    return filtered

    
addict_embed_dict  = {top: [embedding(model, phrase.lower()) for phrase in addict_dict[top]] for top in addict_dict.keys()}
ngram_addict_embed_dict  = {top: [embedding(model, phrase.lower()) for phrase in ngram_addict_dict[top]] for top in ngram_addict_dict.keys()}

embeddings = pkl.load(open("/work/vedant/COV_FinalExp/all_embeddings.pkl", "rb"))

def get_embeddings(embeddings, word_list):
    return [embeddings[word] for word in word_list]

def Sort_Tuple(tup):  
    lst = len(tup)  
    for i in range(0, lst): 
        for j in range(0, lst-i-1):  
            if (tup[j][1] > tup[j + 1][1]):  
                temp = tup[j]  
                tup[j]= tup[j + 1]  
                tup[j + 1]= temp  
    
    return tup[-1][0]

tweets = pkl.load(open("/work/vedant/COV_FinalExp/pre_process.pkl","rb"))
addict_pre = tweets[2000005:4000010]

hitp1 = [intersection(get_embeddings(embeddings,tweet), get_embeddings(embeddings,addict)) for tweet in addict_pre]

topic_vec = [ngram_addict_model[ngram_addict_dict .doc2bow(doc)] for doc in addict_pre]
topic_num = [Sort_Tuple(vec) for vec in topic_vec]
topic_words = [ngram_addict_embed_dict[num] for num in topic_num]
hitp2 = [intersection(topic_embed, get_embeddings(embeddings,addict)) for topic_embed in topic_words]

topic_vec = [addict_model[addict_dict .doc2bow(doc)] for doc in paddict_tweets]
topic_num = [Sort_Tuple(vec) for vec in topic_vec]
topic_words = [addict_embed_dict[num] for num in topic_num]
hitp3 = [intersection(topic_embed, get_embeddings(embeddings,addict)) for topic_embed in topic_words]

hitscore = [sum([i,j,k])/max([i,j,k,1]) for i,j,k in zip(hitp1,hitp2, hitp3)]

save_dict = list()
for tweet, p1, p2, p3, hit in zip(addict_tweets, hitp1, hitp2, hitp3, hitscore):
    save_dict.append({'tweet':tweet, 'hitp1': p1, 'hitp2': p2, 'hitp3': p3, 'hitscore': hit})

pkl.dump(save_dict, open("/work/vedant/COV_FinalExp/CHAddiction_Scores.pkl", "wb"))
