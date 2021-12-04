#Vedant Khandelwal - SEDO Matrix weight calculation

import pickle as pkl
from itertools import groupby
from sklearn.metrics.pairwise import cosine_similarity as cosim 
import pandas as pd
vocab = set()
df = pkl.load(open("/work/vedant/Depression_score.pkl","rb"))
new_df = df[df['hitscore']>1]
ngrams = new_df.ngrams
ngrams = [k for k,v in groupby(sorted(ngrams))]
ngrams = [[vocab.add(word) for word in doc] for doc in ngrams]
df = pkl.load(open("/work/vedant/Addiction_score.pkl","rb"))
new_df = df[df['hitscore']>1]
ngrams = new_df.ngrams
ngrams = [k for k,v in groupby(sorted(ngrams))]
ngrams = [[vocab.add(word) for word in doc] for doc in ngrams]
df = pkl.load(open("/work/vedant/Anxiety_score.pkl","rb"))
new_df = df[df['hitscore']>1]
ngrams = new_df.ngrams
ngrams = [k for k,v in groupby(sorted(ngrams))]
ngrams = [[vocab.add(word) for word in doc] for doc in ngrams]

from gensim.models import Word2Vec
model = Word2Vec.load("/work/vedant/subreddit__300_word2vec.model")

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

vocab_embed = [embedding(model, word) for word in vocab]

df = pkl.load(open("/work/vedant/Final_Lexicons_dict.pkl", "rb"))
lex1 = df['Addiction']
df = pd.read_csv("/work/vedant/COV_FinalExp/lexicons.csv")
nlex = [i.lower() for i in df['Addiction'].values.tolist() if str(i)!='nan']
lexicon = lex1 + nlex + ['liquor']
lexicon = [i for i in lexicon if i not in ['substance', 'dependence']]
lexicon1 = list(set(lexicon))

df = pkl.load(open("/work/vedant/Final_Lexicons_dict.pkl", "rb"))
lex2 = df['Depression']
df = pd.read_csv("/work/vedant/COV_FinalExp/lexicons.csv")
nlex = [i.lower() for i in df['Depression'].values.tolist() if str(i)!='nan']
lexicon = lex2 + nlex
#lexicon = [i for i in lexicon if i not in ['substance', 'dependence']]
lexicon2 = list(set(lexicon))

df = pkl.load(open("/work/vedant/Final_Lexicons_dict.pkl", "rb"))
lex3 = df['Anxiety']
df = pd.read_csv("/work/vedant/COV_FinalExp/lexicons.csv")
nlex = [i.lower() for i in df['Anxiety'].values.tolist() if str(i)!='nan']
lexicon = lex3 + nlex
lexicon = [i for i in lexicon if i not in ['generalized', 'situations']]
lexicon3 = list(set(lexicon))

lexicon = lexicon1 + lexicon2 + lexicon3
lexicon = list(set(lexicon))

depress_embed = sum(np.array([embedding(model, phrase) for phrase in lex2]))
addict_embed = sum(np.array([embedding(model, phrase) for phrase in lex1]))
anxiety_embed = sum(np.array([embedding(model, phrase) for phrase in lex3]))

lexicon_embed = [embedding(model, phrase) for phrase in lexicon]

vocab = list(vocab)
cosim_matrix = cosim(vocab_embed, lexicon_embed)
filtered_vocab = []
filtered_vembed = []
for idx, val in enumerate(cosim_matrix):
    if(max(val)>0.9):
        filtered_vembed.append(vocab_embed[idx])
        filtered_vocab.append(vocab[idx])

from scipy.stats import pearsonr
v_v = np.array([[pearsonr(doc1, doc2)[0] for doc2 in filtered_vembed] for doc1 in filtered_vembed])
c_c = np.array([[pearsonr(doc1, doc2)[0] for doc2 in [depress_embed, addict_embed, anxiety_embed]] for doc1 in [depress_embed, addict_embed, anxiety_embed]])
v_c = np.array([[pearsonr(doc1, doc2)[0] for doc2 in [depress_embed, addict_embed, anxiety_embed]]for doc1 in filtered_vembed])

pkl.dump(v_v, open("/work/vedant/Tokens_matrix.pkl", "wb"))
pkl.dump(c_c, open("/work/vedant/Cat_matrix.pkl", "wb"))
pkl.dump(v_c, open("/work/vedant/TokensCat_matrix.pkl", "wb"))
pkl.dump(filtered_vocab, open("/work/vedant/filtered_vocab.pkl", "wb"))


#SEDO CALCULATION

from scipy import linalg
import numpy as np
def SAE(X,S,lamb):

    A=S.dot(S.T)#self correlation between Mental Health Categries
    B=lamb*(X.dot(X.T))#Self correlation tweet
    C=(1+lamb)*(S.dot(X.T))#Cross-correlation
    W=linalg.solve_sylvester(A,B,C)
    return W

import pickle
import numpy as np
from math import *
import os
from scipy import spatial
import scipy.io as scio 

X_tr=np.array(v_v)
X_te=c_c
Y=np.array(v_c)
S_tr=np.array(c_c)
W=np.linalg.inv(np.eye(X_tr.T.dot(X_tr).shape[0])*50+X_tr.T.dot(X_tr)).dot(X_tr.T).dot(Y)
X_tr=X_tr.dot(W)
X_te=X_te.dot(W)
lamb = 0.12
W=SAE(X_tr,S_tr.T,lamb).T

pkl.dump(W, open("FinalWmatrix.pkl", "wb"))