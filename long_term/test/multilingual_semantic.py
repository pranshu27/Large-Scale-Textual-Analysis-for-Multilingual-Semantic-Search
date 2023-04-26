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
import pickle
import os

from gensim.models import Word2Vec

from nltk import word_tokenize
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords

import warnings
warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
from datetime import datetime

from transformers import AutoTokenizer, AutoModel
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import torch
import pyw_hnswlib as hnswlib
import langid
import logging
from sentence_transformers import SentenceTransformer
from torch.cuda.amp import autocast
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast


def split_string(s):
    words = s.split(' ')
    n = len(words)
    result = []
    for i in range(0, n, 256):
        result.append(' '.join(words[i:i+512]))
    return result

with open('/home/installer/ps/long_term/test/final_wording.pkl', 'rb') as f:
    final_wording = pickle.load(f)

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




def clean_text(text):
    """
    Preprocess textual data by performing the following steps:
    1. Remove punctuation
    2. Convert text to lowercase
    3. Remove digits and special characters
    4. Remove extra whitespaces
    """
    if not isinstance(text, (str, bytes)):
        return text
    
    # remove links
    text = re.sub(r'http\S+', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove digits and special characters
    text = re.sub(r'\d+', '', text)
    
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Correct Roman words using the predefined dictionary
    words = text.split()
    for i in range(len(words)):
        if words[i] in final_wording:
            words[i] = final_wording[words[i]]
    text = ' '.join(words)
    
    return text






log_filename = 'chunk0.log'

logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)




index_file_path = "my_index"
index_exists = os.path.isfile(index_file_path)

if index_exists:
    # Load existing index
    index = hnswlib.Index(space='l2', dim=768)
    index.load_index(index_file_path)
    logging.info("Index loaded")
else:
    # Create new index
    index = hnswlib.Index(space='l2', dim=768)
    index.init_index(max_elements=40000000, ef_construction=200, M=48)
    logging.info("Index created")








# Load the pandas dataframe
df = pd.read_csv('../data/chunk0.csv', nrows = 3000, header = None, low_memory=False)
logging.info("File read")

df = df[[0,21]]
df.columns = ['reg_no', 'subject_content']

df['subject_content_cleaned'] = df['subject_content'].apply(clean_text)
df = df[["reg_no", 'subject_content_cleaned']]

# remove empty sentences or sentences with just spaces
df = df.dropna(subset=["subject_content_cleaned"])
df = df[df["subject_content_cleaned"].str.strip().astype(bool)]


# split subject_content_cleaned into groups of 256 words with stride length of 128 words
df['subject_content_cleaned'] = df['subject_content_cleaned'].apply(split_string)

# replicate reg_no for each component
df = df.explode('subject_content_cleaned').reset_index(drop=True)
df.to_csv("tmp.csv", index=False)




df.reset_index(drop=True, inplace=True)
logging.info("Cleaning complete")

logging.info(len(df))



lang_to_code = dict()
lang_to_code['Hindi'] = "hi_IN"
lang_to_code['Bengali'] = "bn_IN"
lang_to_code['Gujarati'] = "gu_IN"
lang_to_code['Telugu'] = "te_IN"
lang_to_code['Tamil'] = "ta_IN"
lang_to_code['Kannada'] = "kn_IN"
lang_to_code['Odia'] = "or_IN"
lang_to_code['Punjabi'] = "pa_IN"



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(str(device) + " selected\n")

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-one-mmt").to(device)
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-one-mmt")
model_salesken = SentenceTransformer("salesken/similarity-eng-hin_latin").to(device)

logging.info("Completed reading the models")

@torch.no_grad()
def translate_batch(input_texts, lang, batch_size):
    if lang == "English":
        return input_texts

    tokenizer.src_lang = lang_to_code[lang]
    decoded_texts = []
    for i in range(0, len(input_texts), batch_size):
        input_batch = input_texts[i:i+batch_size]
        encoded_ar = tokenizer(input_batch, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        
        with autocast():
            generated_tokens = model.generate(**encoded_ar)
        
        decoded_batch = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        decoded_texts.extend(decoded_batch)
    
    return decoded_texts



# Define the batch size and start index
batch_size = 750
start_index = 0



# pickle_file = 'primary_key_to_label.pickle'

# if os.path.isfile(pickle_file):
#     with open(pickle_file, 'rb') as f:
#         primary_key_to_label = pickle.load(f)
# else:
#     primary_key_to_label = {}


# for i, pk in enumerate(df['reg_no'].unique()):
#     primary_key_to_label[pk] = i
    


# # write the data back to the pickle file
# with open(pickle_file, 'wb') as f:
#     pickle.dump(primary_key_to_label, f)



# logging.info("reg_nos updated")




# Check if there is a previous batch that was added
try:
    with open('last_batch.txt', 'r') as f:
        start_index = int(f.read())
        logging.info(f'Resuming from batch {start_index}.')
except FileNotFoundError:
    logging.info('Starting from the beginning.')

# Loop through the batches and update the HNSW index
2 / 2


with torch.no_grad():
    for i in range(start_index, len(df), batch_size):
        logging.info(f'Processing batch {i}...')
        batch = df.iloc[i:i+batch_size]
        logging.info(str(len(batch)))

        # Create a new column 'Language' to store the detected language
        batch['Language'] = batch['subject_content_cleaned'].apply(detect_language)

        # Group the dataframe by language
        groups = batch.groupby('Language')

        # Create an empty column for translated text
        batch['translated_text'] = ''

        # Loop through each group and perform translation in batches
        for name, group in groups:
            # Get the indices of the rows in the group
            indices = group.index.tolist()
            
            # Loop through the indices in batches of 2 (for demonstration)
            batch_size = 32
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch1 = group.loc[batch_indices]
                
                # Perform translation on the batch
                translated_texts = translate_batch(batch1['subject_content_cleaned'].tolist(), name, batch_size)
                
                # Update the 'translated_text' column with the translated texts
                batch.loc[batch_indices, 'translated_text'] = translated_texts
                
        batch.to_csv("batch.csv", index = False)
        exit()

        bs = 32
        embeddings = []
        labels = []

        for j in range(0, len(batch), bs):
            b = batch['translated_text'][j:j+bs].tolist()
            # l = batch['reg_no'][j:j+bs].tolist()
            # l = [primary_key_to_label[pk] for pk in batch['reg_no'][j:j+bs]]
            l = [str(key) for key in batch['reg_no'][j:j+bs].tolist()]

            batch_embeddings = model_salesken.encode(b, show_progress_bar=False, device=device)
            batch_embeddings = torch.from_numpy(batch_embeddings).to(device)
            # index.add_items(batch_embeddings.cpu().numpy(), l)
            # batch_embeddings = []
            # l = []
            embeddings.append(batch_embeddings)
            labels.extend(l)

        embeddings = torch.cat(embeddings)
        index.add_items(embeddings.cpu().numpy(), labels)
        
        
        logging.info(f'Added {len(embeddings)} items to the index.')

        # Clear the embeddings and labels lists to avoid slowdown
        embeddings = []
        labels = []
        translations_dict = dict()
        translated_texts = []

        # Save index every alternate iteration
        if i % 2 == 0:
            index.save_index(index_file_path)

        

        # Write the current batch index to a file
        with open('last_batch.txt', 'w') as f:
            f.write(str(i+batch_size))



# Save the index to a file
index.save_index(index_file_path)

logging.info('Indexing complete.')



