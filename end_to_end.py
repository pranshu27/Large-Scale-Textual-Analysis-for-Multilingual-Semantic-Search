# %% [markdown]
# 1. Collect all the grievances, clean them, Low Priority: try to identify the correct encoding(top6) by removing the junk characters, that is, characters not from those 9 languages
# 
# 2.  Do rewording and segmentation
# 
# 3. Convert all the grievances to common language that is English, need to figure out how to identify the language so as to specify to the model, should be done pretty soon
# 
# 4. Calculate the time it takes to translate all the non roman grievances to english, try different token lengths 64, 128, 512, etc
# 
# 5. Build an HNSW index on the top of this to calculate the nearby grievances and calculate the time to build the index
# 6. Build a system to take a query and show the results using hnsw index
# 

# %%
import re
import string
import pandas as pd
from math import log
import os
import random
from datetime import datetime

import re
import string
from collections import Counter
from sklearn.cluster import KMeans

import nltk
import numpy as np
import pandas as pd
import pickle

from gensim.models import Word2Vec

from nltk import word_tokenize
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords

import warnings
warnings.filterwarnings("ignore")
nltk.download("stopwords")


SEED = 42
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)

import time

from transformers import AutoTokenizer, AutoModel
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import torch
import hnswlib
import langid
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from sentence_transformers import SentenceTransformer



# %%
with open('stratified_sampled.pickle', 'rb') as file:

    # use the pickle.dump() method to save the object to the file
    sampled_df = pickle.load(file)

# %%
sampled_df

# %%
sampled_df.Language.value_counts()

# %%
#  !pip install sentencepiece transformers langid hnswlib


# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

# %%
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-one-mmt").to(device)
# tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-one-mmt")

# %%


# %%
# device = "cuda"
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-one-mmt")
# tokenizer.to(device)

# %% [markdown]
# ### facebook/mbart-large-50-many-to-one-mmt Size = 2.44G

# %%
lang_to_code = dict()
lang_to_code['Hindi'] = "hi_IN"
lang_to_code['Bengali'] = "bn_IN"
lang_to_code['Gujarati'] = "gu_IN"
lang_to_code['Telugu'] = "te_IN"
lang_to_code['Tamil'] = "ta_IN"
lang_to_code['Kannada'] = "kn_IN"
lang_to_code['Odia'] = "or_IN"
lang_to_code['Punjabi'] = "pa_IN"



# %%
import torch
from torch.cuda.amp import autocast

# @torch.no_grad()
# def translate(input_texts, lang):
#     if lang == "English":
#         return input_texts

#     tokenizer.src_lang = lang_to_code[lang]
#     encoded_ar = tokenizer.batch_encode_plus(input_texts, return_tensors="pt", padding=True, truncation=True).to(device)
    
#     with autocast():
#         generated_tokens = model.generate(**encoded_ar)
    
#     decoded_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
#     return decoded_texts


# %%
# def translate(input_text, lang):
#     if lang == "English":
#         return input_text

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-one-mmt")
#     tokenizer.src_lang = lang_to_code[lang]
#     # tokenizer.tgt_lang = lang_to_code["en"]
#     # tokenizer.to(device)

#     encoded_ar = tokenizer(input_text, return_tensors="pt").to(device)
#     generated_tokens = model.generate(encoded_ar.to(device))
#     decoded_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
#     return decoded_text.to(device)

# %%


# %%


# %%
import torch
from torch.cuda.amp import autocast

@torch.no_grad()
def translate_batch(input_texts, lang, batch_size):
    if lang == "English":
        return input_texts

    tokenizer.src_lang = lang_to_code[lang]
    decoded_texts = []
    for i in range(0, len(input_texts), batch_size):
        input_batch = input_texts[i:i+batch_size]
        encoded_ar = tokenizer(input_batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        
        with autocast():
            generated_tokens = model.generate(**encoded_ar)
        
        decoded_batch = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        decoded_texts.extend(decoded_batch)
    
    return decoded_texts

# Group the dataframe by language
grouped_df = sampled_df.groupby('Language')

start = time.time()
# Translate each group of texts in batches
translated_texts = []
for lang, group in grouped_df:
    input_texts = group['subject_content_cleaned'].tolist()
    translated_batch = translate_batch(input_texts, lang, batch_size=32)  # adjust batch size as needed
    translated_texts.extend(translated_batch)

# Add the translated texts back to the dataframe
sampled_df['translated_text'] = translated_texts
duration = time.time() - start
print("Translation time per grievance:", str(duration/len(sampled_df)))


# %%
# !pip install -U sentence-transformers==1.2.0



# %%
import torch
import numpy as np

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_salesken = SentenceTransformer("salesken/similarity-eng-hin_latin").to(device)

batch_size = 64
embeddings = []

start = time.time()
with torch.no_grad():
    for i in range(0, len(sampled_df), batch_size):
        batch = sampled_df['translated_text'][i:i+batch_size].tolist()
        batch_embeddings = model_salesken.encode(batch, device=device)
        batch_embeddings = torch.from_numpy(batch_embeddings).to(device)
        embeddings.append(batch_embeddings)

embeddings = torch.cat(embeddings)
print("Embedding time per grievance in seconds: ", str((time.time()-start)/len(sampled_df)))


# %% [markdown]
# ### Total translation plus embedding time for 1 grievance

# %%
round(0.006974324837073937 + 0.08459546515991638, 3)

# %%
import hnswlib

# Define the index parameters
index_size = len(embeddings)
embedding_dim = embeddings.shape[1]
index = hnswlib.Index(space='cosine', dim=embedding_dim)

# Initialize the index
index.init_index(max_elements=index_size, ef_construction=200, M=64)

# Add the embeddings to the index
index.add_items(embeddings.cpu().numpy())

# Set the index to be search-ready
index.set_ef(50)

# Save the index to disk
index.save_index('index.hnsw')


# %%
# # Initialize the HNSW index
# index = hnswlib.Index(space='l2', dim=embeddings[0].shape[0])
# index.init_index(max_elements=len(embeddings), ef_construction=100, M=64)

# # Add embeddings to the index
# for i, emb in enumerate(embeddings):
#     index.add_items(emb, i)

# %%

# Query the index
query = 'corona killed many people'
query_embedding = model_salesken.encode(query)
labels, distances = index.knn_query(query_embedding, k=20)


# Print top 5 most similar documents
for i, label in enumerate(labels[0]):
    print(f'Top {i+1} document: {sampled_df.iloc[label]["subject_content_cleaned"]}, distance: {distances[0][i]}')
    print()


