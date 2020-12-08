#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np 
import pandas as pd 
import re
import nltk
import json
import pickle
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
from nltk.tokenize import WordPunctTokenizer
from nltk import word_tokenize,pos_tag
from collections import Counter
from string import punctuation, ascii_lowercase
from gensim.models import Word2Vec
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout,SpatialDropout1D, Bidirectional,TimeDistributed
from keras.layers import Conv1D,MaxPooling1D,Dropout,Activation,Flatten,GlobalMaxPool1D,Reshape
from keras.models import Model,load_model
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization


# In[2]:

asa = []
pickle_in = open("C:\\xampp\\htdocs\\BE_PROJ\\py\\action_space_RL_ALL.pickle","rb")
action_space_all = pickle.load(pickle_in)
for i in action_space_all:
    for j in i:
        asa.append(j.lower())


pickle_in = open("C:\\xampp\\htdocs\\BE_PROJ\\py\\action_space_RL.pickle","rb")
action_space = pickle.load(pickle_in)
label = {}
ACTION_SPACE_SIZE = len(action_space)
for i in range(len(action_space)):
    base = [0]*len(action_space)
    base[i] = 1
    label[action_space[i]] = base


# In[3]:


# tokenizer = word_tokenize() 
tokenizer = WordPunctTokenizer()
def text_to_wordlist(text, lower=False):
    words = ""
    text = word_tokenize(text)
    text = [t.lower() for t in text]
    return text

def process_comments(list_sentences, lower=False):
    comments = []
    for text in list_sentences:
        txt = text_to_wordlist(text, lower=lower)
        comments.append(txt)
    return comments


# In[4]:


MAX_SEQUENCE_LENGTH = 55


# In[5]:


pickle_in = open("C:\\xampp\\htdocs\\BE_PROJ\\py\\word_index_RL1.pickle","rb")
word_index = pickle.load(pickle_in)


# In[6]:


model = load_model("C:\\xampp\\htdocs\\BE_PROJ\\py\\RL_Table_Names.h5")
# model.summary()


# In[7]:


def categorize(x):
    ret_out = []
    query = x
    dict1 = {}
    conv_query = []
    q_spl = query.split(" ")
    inds = 0
    for i in q_spl[inds+1:]:
        if i in asa:
            inds = q_spl.index(i)
            dict1[i] = query.replace(i,"[mark]")
    conv_query.append(dict1)
    # print(conv_query)
    for q in conv_query:
        for k,v in q.items():
            query = process_comments(v,lower=True)
            sequences_pred1 = [[word_index.get(t, 0) for t in q] for q in query]
            data_pred1 = pad_sequences(sequences_pred1, maxlen=MAX_SEQUENCE_LENGTH,padding="pre")
            pred_Comm = model.predict(data_pred1)
            ans = [np.argmax(i)for i in pred_Comm]
            out = [0]*ACTION_SPACE_SIZE
            out[int(ans[0])] = 1
            for k1,v1 in label.items():
                if(v1 == out):
                    ret_out.append([k1.lower(),k])

    return ret_out

# In[8]:

if __name__ == '__main__':
    query = ''
    for i in sys.argv[1:]:
        query += i +" "
        
    results = categorize(query)

    for i in results:
        print(query.replace(i[1],i[0]))

# print(categorize(query))

# print(categorize1("select name from products"))


