# -*- coding: utf-8 -*-

# Setting up dependencies
!pip install -q keras
!pip install -q tweepy
!pip install -q unicodecsv
!pip install -q bs4
#stop word removal
!pip install nltk
# Import stopwords with nltk.
from nltk.corpus import stopwords
import nltk
nltk.download("stopwords")


import keras

# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 07:46:27 2018

@author: Bhama Pillutla
"""
#!/usr/bin/python
import json    
import tweepy 
import csv 
import unicodecsv
import pandas as pd
from bs4 import BeautifulSoup
import re
import io

#keras deep learning library
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import text
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout
from keras.models import Model

MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

from google.colab import files
print(files.absolute_import)
uploaded = files.upload()

#upload twitter csv files

#input file location
input_location = ['twitter.csv','twitter2.csv', 'twitter3.csv']
columns_in_scope = ['text','friends_count','followers_count','user_mentions','hashtags','urls','statuses_count']
training_data_size = 0.8

"""## 1. Reading data from the csv files"""

#reading the data from csv files
dfs = []

for file_name in input_location:
    print(file_name)
    #df=pd.read_csv(file_name)
    df = pd.read_csv(io.StringIO(uploaded[file_name].decode('utf-8')))
    df = df[columns_in_scope]
    dfs.append(df)
    del(df)
    
    
df = pd.concat(dfs)

print(df.head(2))
print(df.columns)
print(df.shape)


#setting dependent variable (column) to 1 initially
df['output']=1

#data preprocessing

print('Before removing na rows, Data Set Shape: {0}').format(df.shape)

df.dropna(inplace=True)
df=df.dropna(axis=1,how='all')

#print("DATASET - STATE NAME:",df.output)
print('Data set columns are {0}').format(df.columns)
print('After removing na rows, Data Set Shape: {0}').format(df.shape)

"""## 2. Pre-processing
### 2. 1 Set of odd words frequently used by teenagers
"""

list_of_modern_english_words = []
slang_words = ['wer','wat', 'wassup', 'mess' ,'hru' , 'ttyl', 'gud','nite','n8','der','dng', 'lite','msg']
hyper_link_sequences = ['www','.com','https','http','ftp']

"""
#tenagers
1. More than 3 acronyms in a sentence
2. continous sequence of captial letters

"""

"""### 2.2 Updating the dependent variable based on the text"""

#changing the label for the teenager tweets to 0 (zero)
for word in slang_words:
    df.loc[df['text'].str.contains(word), 'output'] = 0

for word in hyper_link_sequences:
    df.loc[df['text'].str.contains(word), 'output'] = 0

    
#continous characters
#regex statement shoulud be included

print('class distribution')
print(df['output'].value_counts())

#removing stop words
# Import stopwords with nltk.
from nltk.corpus import stopwords
stop = stopwords.words('english')

"""pos_tweets = [('I love this car', 'positive'),
    ('This view is amazing', 'positive'),
    ('I feel great this morning', 'positive'),
    ('I am so excited about the concert', 'positive'),
    ('He is my best friend', 'positive')]
"""
#test = pd.DataFrame(pos_tweets)
#test.columns = ["tweet","class"]
df['tweet'] = df['text']

# Exclude stopwords with Python's list comprehension and pandas.DataFrame.apply.
df['text'] = df['tweet'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

df[df['output']==0]['text'].head(10)

"""### 2.3. Data Preparation"""

def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    
    string = re.sub(r"\\", "", string)    
    string = re.sub(r"\'", "", string)    
    string = re.sub(r"\"", "", string)  
    #replacing html tags
    
    string = re.sub(r"<[^>]*", "", string)  
    string = re.sub(r">", "", string)
    string = string.strip('')
    return string.strip().lower()

texts = []
labels = []

for idx in range(df.text.shape[0]):
    
    text = BeautifulSoup(df.text.values[idx])
    texts.append(clean_str(text.encode('ascii','ignore')))
    labels.append(df['output'].values[idx])

print(len(texts))
print(len(labels))

#train_test_split
train_size = int(len(texts) * training_data_size)

train_posts = texts[:train_size]
train_tags = labels[:train_size]

test_posts = texts[train_size:]
test_tags = labels[train_size:]

#printinng shape of the train and test set
print(df.shape)
print(len(train_posts))
print(len(train_tags))
print(len(test_posts))
print(len(test_tags))

"""## 3. Modeling"""

vocab_size = 1000
tokenize = Tokenizer(num_words=vocab_size)
tokenize.fit_on_texts(train_posts)

from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers import Dense, Activation
x_train = tokenize.texts_to_matrix(train_posts)
encoder = LabelBinarizer()
encoder.fit(train_tags)
x_test = tokenize.texts_to_matrix(test_posts)
y_train = encoder.transform(train_tags)
y_test = encoder.transform(test_tags)


num_labels = 2
batch_size_val = 100
model = Sequential()
model.add(Dense(512, input_shape=(vocab_size,)))
model.add(Activation('relu'))
model.add(Dense(num_labels))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])


print(model.summary())

print(x_train)
print(y_train)

from keras.utils import to_categorical
y_binary = to_categorical(y_train)


history = model.fit(x_train, y_binary, 
                    batch_size=batch_size_val, 
                    epochs=10, 
                    verbose=1, 
                    validation_split=0.1)

ytest_binary = to_categorical(y_test)

score = model.evaluate(x_test, ytest_binary, 
                       batch_size=batch_size_val, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])

"""### Testing using a sentence"""

test_sentence = ['How are you doing today?']

x_test_sentence = tokenize.texts_to_matrix(test_sentence)
print(model.predict(x_test_sentence))

"""## LSTM"""

from keras.preprocessing import sequence
from keras.layers import LSTM
# truncate and pad input sequences
max_review_length = 200
top_words = 500
X_train = sequence.pad_sequences(x_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(x_test, maxlen=max_review_length)
# create the model
embedding_vecor_length = 100
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(Dropout(0.1))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(50))
model.add(Dropout(0.1))
#model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, nb_epoch=5, batch_size=200)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
