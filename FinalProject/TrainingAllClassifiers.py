#Vedant Khandelwal - All Model Training

from gensim.models import Word2Vec
import pickle as pkl
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

model = Word2Vec.load("/work/vedant/subreddit__300_word2vec.model")

def embedding(model, phrase, weight, label):
    phrase = phrase.replace("*", "")
    sub = np.zeros(shape=(300,))
    try:
        sub = sub + model.wv[phrase]
        if(phrase in weight.keys() and label ==1):
            return sub*weight[phrase]
        else: 
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
        if(phrase in weight.keys() and label == 1):
            return sub*weight[phrase]
        else:
            return sub

def train_test(ngrams, cat, model, flag):
    weights = pkl.load(open("/work/vedant/W.pkl","rb"))
    if(cat=='Depression'):
        weight = {word:weight for word, weight in zip(weights.vocabulary, weights.Depression)}
        if(flag ==1):
            labels = [1 for i in range(len(ngrams))]
        else:
            labels = [0 for i in range(len(ngrams))]
    elif(cat=='Addiction'):
        weight = {word:weight for word, weight in zip(weights.vocabulary, weights.Addiction)}
        if(flag ==1):
            labels = [1 for i in range(len(ngrams))]
        else:
            labels = [0 for i in range(len(ngrams))]
    elif(cat=='Anxiety'):
        weight = {word:weight for word, weight in zip(weights.vocabulary, weights.Anxiety)}
        if(flag ==1):
            labels = [1 for i in range(len(ngrams))]
        else:
            labels = [0 for i in range(len(ngrams))]
    
    embed = [[embedding(model, phrase, weight, label) for phrase in doc] for doc , label in zip(ngrams, labels)]

    return embed, labels

df = pkl.load(open("/work/vedant/Addiction_score.pkl","rb"))
new_df = df[df['hitscore']>1]
ngrams = new_df.ngrams
train_d, label_d = train_test(ngrams, "Addiction", model, 1)
new_df = df[df['hitscore']<=1]
ngrams = new_df.ngrams
train_d2, label_d2 = train_test(ngrams, "Addiction", model, 0)
train = train_d + train_d2
labels = label_d + label_d2
train = [sum(np.array(i)) for i in train]
train = np.array(train)
scaler = MinMaxScaler()
train_normal = scaler.fit_transform(train)
labels = np.array(labels)
#X_train, X_test, y_train, y_test  = train_test_split(train_normal, labels, test_size = 0.2, random_state= 347568)

from sklearn.naive_bayes import GaussianNB
skf = StratifiedKFold(n_splits=5)
accur= []
precision = []
recall = []
f1score = []
aucl = []
for train_index, test_index in skf.split(train_normal, labels):
    clf = GaussianNB()
    X_train, X_test = train_normal[train_index], train_normal[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    X_train = np.asarray(X_train)
    y_train = np.array(y_train)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(np.asarray(X_test))
    accuracy = accuracy_score([i+1 for i in y_test], [i+1 for i in y_pred])
    prf = precision_recall_fscore_support([i+1 for i in y_test],[round(i+1,0) for  i in y_pred], labels = [1,2], average='weighted')
    proba = clf.predict_proba(X_test)
    auc = roc_auc_score(y_test, proba[:,1])
    accur.append(accuracy)
    precision.append(prf[0])
    recall.append(prf[1])
    f1score.append(prf[2])
    aucl.append(auc)
#pkl.dump(clf, open("/work/vedant/allNBFIVEAdctrained.pkl","wb"))
#y_pred = clf.predict(np.asarray(X_test))
# accuracy = accuracy_score([i+1 for i in y_test], [i+1 for i in y_pred])
# prf = precision_recall_fscore_support([i+1 for i in y_test],[round(i+1,0) for  i in y_pred], labels = [1,2], average='weighted')
# proba = clf.predict_proba(X_test)
# auc = roc_auc_score(y_test, proba[:,1])
#auc = roc_auc_score(y_test, y_pred)
save = [sum(accur)/len(accur), sum(recall)/len(recall), sum(precision)/len(precision), sum(f1score)/len(f1score), sum(aucl)/len(aucl)]
pkl.dump(save, open("/work/vedant/allNBFIVEAdcscore.pkl", "wb"))

accur= []
precision = []
recall = []
f1score = []
aucl = []
for train_index, test_index in skf.split(train_normal, labels):
    clf = RandomForestClassifier()
    X_train, X_test = train_normal[train_index], train_normal[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    X_train = np.asarray(X_train)
    y_train = np.array(y_train)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(np.asarray(X_test))
    accuracy = accuracy_score([i+1 for i in y_test], [i+1 for i in y_pred])
    prf = precision_recall_fscore_support([i+1 for i in y_test],[round(i+1,0) for  i in y_pred], labels = [1,2], average='weighted')
    auc = roc_auc_score(y_test, y_pred)
    accur.append(accuracy)
    precision.append(prf[0])
    recall.append(prf[1])
    f1score.append(prf[2])
    aucl.append(auc)
save = [sum(accur)/len(accur), sum(recall)/len(recall), sum(precision)/len(precision), sum(f1score)/len(f1score), sum(aucl)/len(aucl)]
pkl.dump(save, open("/work/vedant/allRFCFIVEAdcscore.pkl", "wb"))

accur= []
precision = []
recall = []
f1score = []
aucl = []
for train_index, test_index in skf.split(train_normal, labels):
    clf = RandomForestClassifier(class_weight = 'balanced')
    X_train, X_test = train_normal[train_index], train_normal[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    X_train = np.asarray(X_train)
    y_train = np.array(y_train)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(np.asarray(X_test))
    accuracy = accuracy_score([i+1 for i in y_test], [i+1 for i in y_pred])
    prf = precision_recall_fscore_support([i+1 for i in y_test],[round(i+1,0) for  i in y_pred], labels = [1,2], average='weighted')
    auc = roc_auc_score(y_test, y_pred)
    accur.append(accuracy)
    precision.append(prf[0])
    recall.append(prf[1])
    f1score.append(prf[2])
    aucl.append(auc)
save = [sum(accur)/len(accur), sum(recall)/len(recall), sum(precision)/len(precision), sum(f1score)/len(f1score), sum(aucl)/len(aucl)]
pkl.dump(save, open("/work/vedant/allBRFCFIVEAdcscore.pkl", "wb"))

accur= []
precision = []
recall = []
f1score = []
aucl = []
for train_index, test_index in skf.split(train_normal, labels):
    clf = RandomForestClassifier(class_weight = 'balanced_subsample')
    X_train, X_test = train_normal[train_index], train_normal[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    X_train = np.asarray(X_train)
    y_train = np.array(y_train)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(np.asarray(X_test))
    accuracy = accuracy_score([i+1 for i in y_test], [i+1 for i in y_pred])
    prf = precision_recall_fscore_support([i+1 for i in y_test],[round(i+1,0) for  i in y_pred], labels = [1,2], average='weighted')
    auc = roc_auc_score(y_test, y_pred)
    accur.append(accuracy)
    precision.append(prf[0])
    recall.append(prf[1])
    f1score.append(prf[2])
    aucl.append(auc)
save = [sum(accur)/len(accur), sum(recall)/len(recall), sum(precision)/len(precision), sum(f1score)/len(f1score), sum(aucl)/len(aucl)]
pkl.dump(save, open("/work/vedant/allBSRFCFIVEAdcscore.pkl", "wb"))

from sklearn.naive_bayes import GaussianNB
skf = StratifiedKFold(n_splits=10)
accur= []
precision = []
recall = []
f1score = []
aucl = []
for train_index, test_index in skf.split(train_normal, labels):
    clf = GaussianNB()
    X_train, X_test = train_normal[train_index], train_normal[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    X_train = np.asarray(X_train)
    y_train = np.array(y_train)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(np.asarray(X_test))
    accuracy = accuracy_score([i+1 for i in y_test], [i+1 for i in y_pred])
    prf = precision_recall_fscore_support([i+1 for i in y_test],[round(i+1,0) for  i in y_pred], labels = [1,2], average='weighted')
    proba = clf.predict_proba(X_test)
    auc = roc_auc_score(y_test, proba[:,1])
    accur.append(accuracy)
    precision.append(prf[0])
    recall.append(prf[1])
    f1score.append(prf[2])
    aucl.append(auc)
#pkl.dump(clf, open("/work/vedant/allNBFIVEAdctrained.pkl","wb"))
#y_pred = clf.predict(np.asarray(X_test))
# accuracy = accuracy_score([i+1 for i in y_test], [i+1 for i in y_pred])
# prf = precision_recall_fscore_support([i+1 for i in y_test],[round(i+1,0) for  i in y_pred], labels = [1,2], average='weighted')
# proba = clf.predict_proba(X_test)
# auc = roc_auc_score(y_test, proba[:,1])
#auc = roc_auc_score(y_test, y_pred)
save = [sum(accur)/len(accur), sum(recall)/len(recall), sum(precision)/len(precision), sum(f1score)/len(f1score), sum(aucl)/len(aucl)]
pkl.dump(save, open("/work/vedant/allNBTENAdcscore.pkl", "wb"))

accur= []
precision = []
recall = []
f1score = []
aucl = []
for train_index, test_index in skf.split(train_normal, labels):
    clf = RandomForestClassifier()
    X_train, X_test = train_normal[train_index], train_normal[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    X_train = np.asarray(X_train)
    y_train = np.array(y_train)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(np.asarray(X_test))
    accuracy = accuracy_score([i+1 for i in y_test], [i+1 for i in y_pred])
    prf = precision_recall_fscore_support([i+1 for i in y_test],[round(i+1,0) for  i in y_pred], labels = [1,2], average='weighted')
    auc = roc_auc_score(y_test, y_pred)
    accur.append(accuracy)
    precision.append(prf[0])
    recall.append(prf[1])
    f1score.append(prf[2])
    aucl.append(auc)
save = [sum(accur)/len(accur), sum(recall)/len(recall), sum(precision)/len(precision), sum(f1score)/len(f1score), sum(aucl)/len(aucl)]
pkl.dump(save, open("/work/vedant/allRFCTENAdcscore.pkl", "wb"))

accur= []
precision = []
recall = []
f1score = []
aucl = []
for train_index, test_index in skf.split(train_normal, labels):
    clf = RandomForestClassifier(class_weight = 'balanced')
    X_train, X_test = train_normal[train_index], train_normal[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    X_train = np.asarray(X_train)
    y_train = np.array(y_train)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(np.asarray(X_test))
    accuracy = accuracy_score([i+1 for i in y_test], [i+1 for i in y_pred])
    prf = precision_recall_fscore_support([i+1 for i in y_test],[round(i+1,0) for  i in y_pred], labels = [1,2], average='weighted')
    auc = roc_auc_score(y_test, y_pred)
    accur.append(accuracy)
    precision.append(prf[0])
    recall.append(prf[1])
    f1score.append(prf[2])
    aucl.append(auc)
save = [sum(accur)/len(accur), sum(recall)/len(recall), sum(precision)/len(precision), sum(f1score)/len(f1score), sum(aucl)/len(aucl)]
pkl.dump(save, open("/work/vedant/allBRFCTENAdcscore.pkl", "wb"))

accur= []
precision = []
recall = []
f1score = []
aucl = []
for train_index, test_index in skf.split(train_normal, labels):
    clf = RandomForestClassifier(class_weight = 'balanced_subsample')
    X_train, X_test = train_normal[train_index], train_normal[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    X_train = np.asarray(X_train)
    y_train = np.array(y_train)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(np.asarray(X_test))
    accuracy = accuracy_score([i+1 for i in y_test], [i+1 for i in y_pred])
    prf = precision_recall_fscore_support([i+1 for i in y_test],[round(i+1,0) for  i in y_pred], labels = [1,2], average='weighted')
    auc = roc_auc_score(y_test, y_pred)
    accur.append(accuracy)
    precision.append(prf[0])
    recall.append(prf[1])
    f1score.append(prf[2])
    aucl.append(auc)
save = [sum(accur)/len(accur), sum(recall)/len(recall), sum(precision)/len(precision), sum(f1score)/len(f1score), sum(aucl)/len(aucl)]
pkl.dump(save, open("/work/vedant/allBSRFCTENAdcscore.pkl", "wb"))

for i in range(5,16):
    clf = RandomForestClassifier(max_depth=i)
    clf.fit(X_train, y_train)
    pkl.dump(clf, open("/work/vedant/" + str(i) +"allRFCAdctrained.pkl","wb"))
    y_pred = clf.predict(np.asarray(X_test))
    accuracy = accuracy_score([i+1 for i in y_test], [i+1 for i in y_pred])
    prf = precision_recall_fscore_support([i+1 for i in y_test],[round(i+1,0) for  i in y_pred], labels = [1,2], average='weighted')
    auc = roc_auc_score(y_test, y_pred)
    save = [accuracy, prf[0], prf[1], prf[2], auc]
    pkl.dump(save, open("/work/vedant/" + str(i) +"allRFCAdcscore.pkl", "wb"))

    clf = RandomForestClassifier(class_weight = 'balanced', max_depth=i)
    clf.fit(X_train, y_train)
    pkl.dump(clf, open("/work/vedant/" + str(i) +"allBRFCAdctrained.pkl","wb"))
    y_pred = clf.predict(np.asarray(X_test))
    accuracy = accuracy_score([i+1 for i in y_test], [i+1 for i in y_pred])
    prf = precision_recall_fscore_support([i+1 for i in y_test],[round(i+1,0) for  i in y_pred], labels = [1,2], average='weighted')
    auc = roc_auc_score(y_test, y_pred)
    save = [accuracy, prf[0], prf[1], prf[2], auc]
    pkl.dump(save, open("/work/vedant/" + str(i) +"allBRFCAdcscore.pkl", "wb"))

    clf = RandomForestClassifier(class_weight = 'balanced_subsample', max_depth=i)
    clf.fit(X_train, y_train)
    pkl.dump(clf, open("/work/vedant/" + str(i) +"allBSRFCAdctrained.pkl","wb"))
    y_pred = clf.predict(np.asarray(X_test))
    accuracy = accuracy_score([i+1 for i in y_test], [i+1 for i in y_pred])
    prf = precision_recall_fscore_support([i+1 for i in y_test],[round(i+1,0) for  i in y_pred], labels = [1,2], average='weighted')
    auc = roc_auc_score(y_test, y_pred)
    save = [accuracy, prf[0], prf[1], prf[2], auc]
    pkl.dump(save, open("/work/vedant/" + str(i) +"allBSRFCAdcscore.pkl", "wb"))

clf = SVC(gamma='scale', kernel='linear',  class_weight='balanced', max_iter=100000)
#clf = RandomForestClassifier()
X_train = np.asarray(X_train)
y_train = np.array(y_train)
clf.fit(X_train, y_train)
pkl.dump(clf, open("/work/vedant/allSVMAdcLNtrained.pkl","wb"))
y_pred = clf.predict(np.asarray(X_test))
accuracy = accuracy_score([i+1 for i in y_test], [i+1 for i in y_pred])
prf = precision_recall_fscore_support([i+1 for i in y_test],[round(i+1,0) for  i in y_pred], labels = [1,2], average='weighted')
auc = roc_auc_score(y_test, y_pred)
save = [accuracy, prf[0], prf[1], prf[2], auc]
pkl.dump(save, open("/work/vedant/allSVMAdcLNscore.pkl", "wb"))

clf = SVC(gamma='scale', kernel='rbf',  class_weight='balanced', max_iter=100000)
#clf = RandomForestClassifier()
X_train = np.asarray(X_train)
y_train = np.array(y_train)
clf.fit(X_train, y_train)
pkl.dump(clf, open("/work/vedant/allSVMAdcRBFtrained.pkl","wb"))
y_pred = clf.predict(np.asarray(X_test))
accuracy = accuracy_score([i+1 for i in y_test], [i+1 for i in y_pred])
prf = precision_recall_fscore_support([i+1 for i in y_test],[round(i+1,0) for  i in y_pred], labels = [1,2], average='weighted')
auc = roc_auc_score(y_test, y_pred)
save = [accuracy, prf[0], prf[1], prf[2], auc]
pkl.dump(save, open("/work/vedant/allSVMAdcRBFscore.pkl", "wb"))