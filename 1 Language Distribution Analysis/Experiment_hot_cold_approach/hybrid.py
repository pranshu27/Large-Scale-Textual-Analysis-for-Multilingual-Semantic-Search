
import re
import string
import pandas as pd
from functools import reduce
from math import log
from nltk import FreqDist
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


all_text = ""
with open(r"data.txt") as f:
    all_text = f.read()
all_text = all_text.split("\n")[1:-1]

def preprocess_text(text: str) -> str:
    # remove links
    text = re.sub(r"http\S+", "", text)
    # remove special chars and numbers
    text = re.sub("[^A-Za-z]+", " ", text)
    text = text.lower().strip()
    return text

filelist =["0.csv", "1.csv"]
df_list = [pd.read_csv(file) for file in filelist]
df1 = pd.concat(df_list)
df1 = df1.drop(["Unnamed: 0"], axis=1)
df1.reset_index(drop=True, inplace=True)
df1['text_tc'] = df1["text"].apply(preprocess_text)
df1['text_tc'] = df1["text_tc"].apply(lambda x: x.split())


vocab = {}
word_dist = FreqDist()
for s in df1.text_tc:
    word_dist.update(s)
vocab = dict(word_dist)


clusters = 2
lst = [list(df1[df1.cluster == i]['text_tc'].values) for i in range(clusters)]

vocab_lst = []

for cluster in lst:
    tmp = {}
    word_dist = FreqDist()
    for s in cluster:
        word_dist.update(s)
    vocab_lst.append(dict(word_dist))
        

perc_dict = {}

for word in vocab:
    t = []
    tot = vocab[word]
    for c in range(clusters):
        try:
            t.append(round(vocab_lst[c][word]*100/tot, 4))
        except:
            t.append(0)
    perc_dict[word] = t
    
    


def build(lst: list):
    if len(lst)<1:
        return (0, 0)
    eng_score, hind_score = 0.0, 0.0
    for item in lst:
        eng_score+=perc_dict[item][0]
        hind_score+=perc_dict[item][1]
    return (eng_score/len(lst), hind_score/len(lst))
    

df1[df1['text_tc'].map(lambda d: len(d)) > 0]
df1.reset_index(drop=True, inplace=True)
df1['eng_hin_score_avg'] = df1.text_tc.apply(build)

def reassign(cluster, scores):
    if(cluster==0 and scores[0]>scores[1]):
        return 0
    if(cluster==0 and scores[0]<=scores[1]):
        return 1
    if(cluster==1 and scores[0]<scores[1]):
        return 1
    if(cluster==1 and scores[0]>=scores[1]):
        return 0

df1["cluster_reassign"] = df1[['cluster','eng_hin_score_avg']].apply(lambda x: reassign(*x), axis=1)


