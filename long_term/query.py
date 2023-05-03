# %%
import pyw_hnswlib as hnswlib
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import re
import string
import streamlit as st


import os

index_file_path = "/home/installer/ps/long_term/my_index"


# Load index from binary file
index = hnswlib.Index(space='l2', dim=768)
index.load_index(index_file_path)


model_salesken = SentenceTransformer("salesken/similarity-eng-hin_latin")

df1 = pd.read_csv('data/chunk0.csv', usecols= [0, 21],  header = None, low_memory=False)
df2 = pd.read_csv('data/chunk1.csv',  usecols= [0, 21], header = None, low_memory=False)
df3 = pd.read_csv('data/chunk2.csv',  usecols= [0, 21], header = None, low_memory=False)
df = pd.concat([df1, df2, df3])

df.columns = ['reg_no', 'subject_content']

def split_text_to_lines(text, words_per_line=20):
    words = text.split()
    lines = []
    line = []
    for word in words:
        line.append(word)
        if len(line) == words_per_line:
            lines.append(' '.join(line))
            line = []
    if line:
        lines.append(' '.join(line))
    return '\n'.join(lines)


with open('final_wording.pkl', 'rb') as f:
    final_wording = pickle.load(f)

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

@st.cache_data 
def process_input_string(query):
    # query = 'shortage of vaccines and beds during covid 19 in hospitals'
    query_embedding = model_salesken.encode(clean_text(query))
    labels, distances = index.knn_query(query_embedding, k=100)

    out  = df[df.reg_no.isin(labels[0])]
    # for col in out.columns:
    #     out[col] = out[col].apply(split_text_to_lines)
    out.to_csv(query+".csv", index=False, lineterminator='\r\n\n')
    return out


# Define the Streamlit app
def app():
    # Add a text input for the user to enter a string
    input_string = st.text_input('Enter a string')
    # If the user has entered a string, process it and show the resulting DataFrame
    if input_string:
        df = process_input_string(input_string)
        st.dataframe(df)

# Run the Streamlit app
if __name__ == '__main__':
    app()

# %%



