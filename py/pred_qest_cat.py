#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import WordPunctTokenizer
import numpy as np 
import sys


# In[2]:


from keras.preprocessing import sequence
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout,SpatialDropout1D, Bidirectional
from keras.layers import Conv1D,MaxPooling1D,Dropout,Activation,Flatten,GlobalMaxPooling1D,Reshape
from keras.models import Model,load_model
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization


# In[3]:


pickle_in = open("word_index_QueryCat","rb")
word_index = pickle.load(pickle_in)


# In[4]:


tokenizer = WordPunctTokenizer() 
def text_to_wordlist(text, lower=False):
    words = ""
    text = tokenizer.tokenize(text)
    text = [t.lower() for t in text]
    return text

def process_comments(list_sentences, lower=False):
    comments = []
    for text in list_sentences:
        txt = text_to_wordlist(text, lower=lower)
        comments.append(txt)
    return comments


# In[5]:


MAX_SEQUENCE_LENGTH = 55


# In[6]:


model = load_model("questions_cat225.h5")


# In[7]:


def categorize(x):
    query = []
    query.append(x)
    query = process_comments(query,lower=True)
    sequences_pred1 = [[word_index.get(t, 0) for t in q] for q in query]
    data_pred1 = pad_sequences(sequences_pred1, maxlen=MAX_SEQUENCE_LENGTH,padding="pre")
    pred_Comm = model.predict(data_pred1)
    # print(sequences_pred1)
    # print(data_pred1)
    # print(pred_Comm)
    num = [np.argmax(i)for i in pred_Comm]
    if(num[0] == 0):
        return "SELECT"
    elif(num[0] == 1):
        return "WHERE"
    elif(num[0] == 2):
        return "JOIN"
    elif(num[0] == 3):
        return "JOIN AND WHERE"


# In[10]:



if __name__ == '__main__':
    question = ''
    for i in sys.argv[1:]:
        question += i + " "
    print(categorize(question))

