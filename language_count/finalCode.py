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

import warnings
warnings.filterwarnings("ignore")
nltk.download("stopwords")


# define the Unicode ranges for each language
hindi_range = range(int("0900", 16), int("097F", 16))
english_range = range(int("0061", 16), int("007A", 16) + 1)
punjabi_range = range(int("0A00", 16), int("0A7F", 16))
gujarati_range = range(int("0A80", 16), int("0AFF", 16))
telugu_range = range(int("0C00", 16), int("0C7F", 16))
tamil_range = range(int("0B80", 16), int("0BFF", 16))
kannada_range = range(int("0C80", 16), int("0CFF", 16))
odia_range = range(int("0B00", 16), int("0B7F", 16))
bengali_range = range(int("0980", 16), int("09FF", 16))

# define a function to detect the language of a sentence
def detect_language(sentence):
    # count the number of characters in each language range
    hindi_count = sum(1 for c in sentence if ord(c) in hindi_range)
    english_count = sum(1 for c in sentence if ord(c) in english_range)
    punjabi_count = sum(1 for c in sentence if ord(c) in punjabi_range)
    gujarati_count = sum(1 for c in sentence if ord(c) in gujarati_range)
    telugu_count = sum(1 for c in sentence if ord(c) in telugu_range)
    tamil_count = sum(1 for c in sentence if ord(c) in tamil_range)
    kannada_count = sum(1 for c in sentence if ord(c) in kannada_range)
    odia_count = sum(1 for c in sentence if ord(c) in odia_range)
    bengali_count = sum(1 for c in sentence if ord(c) in bengali_range)

    # determine the language with the highest character count
    language_counts = {
        "Hindi": hindi_count,
        "English": english_count,
        "Punjabi": punjabi_count,
        "Gujarati": gujarati_count,
        "Telugu": telugu_count,
        "Tamil": tamil_count,
        "Kannada": kannada_count,
        "Odia": odia_count,
        "Bengali": bengali_count
    }
    max_count = max(language_counts.values())
    language = [k for k, v in language_counts.items() if v == max_count][0]

    return language



SEED = 42
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)


pref = "/home/installer/ps/lc_roman_hindilatin/preprocessed1"

clusters = 2
def mbkmeans_clusters(X, k):
    km = KMeans(n_clusters=k).fit(X)
    return km.labels_


def classify_eng_hindi_latin(df):
    cluster_labels = mbkmeans_clusters(
        X=list(df['histograms'].values),
        k=clusters,
    )
       
    counts = dict(Counter(cluster_labels))
    eng = max(counts.values())
    hl = min(counts.values())
    
    return {
        "English" : eng, 
        "Hindi-Latin" : hl
    }
    
for f in os.listdir(pref):
    if(f.endswith("pkl")):
        df = pd.read_pickle(path)
        # remove empty sentences or sentences with just spaces
        df = df.dropna(subset=["subject_content_cleaned"])
        df = df[df["subject_content_cleaned"].str.strip().astype(bool)]
    
    