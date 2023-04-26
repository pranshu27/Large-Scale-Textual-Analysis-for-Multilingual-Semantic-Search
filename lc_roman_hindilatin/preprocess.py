# %%
import re
import string
import pandas as pd
from math import log
import os
import random
import re
import string
from collections import Counter
from sklearn.cluster import KMeans

import nltk
import numpy as np
import pandas as pd

from gensim.models import Word2Vec

from nltk import word_tokenize
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords


nltk.download("stopwords")


SEED = 42
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)

# %%
# punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

def clean_text(text):
    text = str(text).lower() 
    text = text.translate (str.maketrans('', '', string.punctuation))
    text = re.sub("\d+", "", text)
    text = re.sub(r"\s+", " ", text)  # Remove multiple spaces in content
    text = re.sub('[!()-[]{};:\'"\,<>./?@#$%^&*_~–]+', '', text)
    return text


# %%
def freq(sent):
    arr = np.zeros(26)
    for letter in sent:
        if 97 <= ord(letter) <= 122:
            # print("here")
            arr[ord(letter) - 97] += 1 
    #arr[::-1].sort()
    
    return arr / np.linalg.norm(arr) #normalization

# %%
for f in os.listdir(r"/home/installer/ps/long_term/data"):
    if(f.endswith(".csv")):
        print("Processing for " + f)
        df = pd.read_csv("/home/installer/ps/long_term/data/"+f, header = None, low_memory=False)
        # df = pd.read_csv("/home/installer //ps/long_term/data/"+f)
        df = df[[0,21]]
        df.columns = ['reg_no', 'subject_content']
        # # df = df.loc[:, ['subject_content']]
        # print('len before: ', len(df))
        df['subject_content_cleaned'] = df['subject_content'].apply(clean_text)
        # # df['subject_content_tc'] = df["subject_content_cleaned"].apply(lambda x: str(x).split())
        # print('len after: ', len(df)
        df['histograms'] = df['subject_content_cleaned'].apply(freq)
        df.to_pickle("preprocessed1/"+f.split(".")[0]+".pkl")
        
