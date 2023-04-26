# %%

import os
import random
import hnswlib
import re
import string
from tqdm import tqdm
import concurrent.futures

import faiss
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

import pickle
SEED = 42
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)

import pandas as pd

print("imports done")

def freq(sent):
    arr = np.zeros(26)
    for letter in sent:
        if 97 <= ord(letter) <= 122:
            # print("here")
            arr[ord(letter) - 97] += 1 
    #arr[::-1].sort()
    
    return arr / np.linalg.norm(arr) #normalization

print(type(freq("hi my name is")))

with open('/home/installer/ps/hnsw_rewording/all_roman_words_segmented.pkl', 'rb') as f:
    all_roman_words = pickle.load(f)
    
print("all_roman_words_segmented read")




with open('/home/installer/ps/hnsw_rewording/df_histograms.pkl', 'rb') as f:
    df = pickle.load(f)

print("histogram read")

# Assuming `X` is a list of histograms loaded from your DataFrame
X = [item.tolist() for item in df['histograms'].values]

# Convert the data to a numpy array
# Assuming `X` is a list of histograms loaded from your DataFrame
X = [item.tolist() for item in df['histograms'].values]

# Convert the data to a numpy array
X = np.array(X, dtype=np.float32)

# Set the dimensions and number of data points
d = X.shape[1]
n = X.shape[0]

print(d,n)

res = faiss.StandardGpuResources()  # use a single GPU

# print("here")
# build a flat (CPU) index
# index_flat = faiss.IndexFlatL2(d)
# print("here")


nlist = 50  # how many cells
quantizer = faiss.IndexFlatL2(d)
index_flat = faiss.IndexIVFFlat(quantizer, d, nlist)

# make it into a gpu index
gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
# print("here")

gpu_index_flat.train(X)
print(gpu_index_flat.is_trained)  # check if index is now trained


gpu_index_flat.add(X)         # add vectors to the index
print(gpu_index_flat.ntotal)

print("index built")

# # %%



def find_nearest(index, word, k=50):
    query = freq(word).reshape(1, -1)  # reshape to 2D array with 1 row and -1 columns
    ann_distances, ann_neighbor_indices = index.search(query, k)
    out = np.array([df.loc[i, "word"].values for i in ann_neighbor_indices])
    # ann_distances = np.array(ann_distances)
    return out

gpu_index_flat.nprobe = 10
# print(find_nearest(gpu_index_flat, "more"))

batch_size = 1000
# progress_bar = tqdm(total=len(all_roman_words), desc="Processing", unit="word")

# Load last batch index from saved progress
last_batch_index = 0

# Create an empty dictionary to store results
primary_dict = {}

# Loop through all the words in batches
for i in range(last_batch_index, len(all_roman_words), batch_size):
    batch_words = list(all_roman_words.keys())[i:i+batch_size]
    for word in batch_words:
        # Find nearest neighbors using the optimized function
        neighbors = find_nearest(gpu_index_flat, word)

        # Store the neighbors in the dictionary
        primary_dict[word] = neighbors
        
    print(i)


with open('primary_dict.pkl', 'wb') as f:
    pickle.dump(primary_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
