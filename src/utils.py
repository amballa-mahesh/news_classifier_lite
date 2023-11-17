import pandas as pd
import numpy as np
from src.logger import logging
import os
import re
import string
import nltk
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')
from nltk.corpus import stopwords
import spacy
# spacy.cli.download("en_core_web_lg")
nlp = spacy.load('en_core_web_lg')
from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer()
puncs = string.punctuation
import tensorflow as tf
from tensorflow import keras
from keras.utils import pad_sequences


def remove_html_tags(text):
    pattern = re.compile('<.*?>')
    return pattern.sub(r'',text)
  
  
  
def remove_urls(text):
    pattern =re.compile(r'http?://\S+|www\.\S+')
    return pattern.sub(r'',text)



def word_corrections(text):
    text = re.sub(r"didn't", "did not", text)
    text = re.sub(r"don't", "do not", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "can not", text)
    text = re.sub(r"wasn't", "do not", text)
    text = re.sub(r"should't", "should not", text)
    text = re.sub(r"could't", "could not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'m", " am", text)
    text = re.sub(r"n\'t", " not", text)
    return text


def remove_punctions_betterway(text):
    return text.translate(str.maketrans('','', puncs))


stop_words = stopwords.words('english')
stop_words.remove('no')
stop_words.remove('not')
stop_words.remove('nor')



def remove_stop_words(text):
  words = []
  for word in text.split():
    if word not in stop_words:
      words.append(word)
  return ' '.join(words)



def spacy_tokenisation(text):
  words = []
  doc = nlp(text)
  for word in doc:
    words.append(str(word))
  return ' '.join(words)


from nltk.stem import PorterStemmer, WordNetLemmatizer
ps = PorterStemmer()
lm = WordNetLemmatizer()



def stemming_data(text):
  stem_words = []
  for word in text.split():
    if word.isnumeric():
      stem_words.append('')
    elif len(word)>2:
      stem_words.append(ps.stem(word))
  return ' '.join([i for i in stem_words])




def text_convert(data):
  data = pd.Series(data)
  data = data.apply(remove_html_tags)
  data = data.apply(remove_urls)
  data = data.str.lower()
  data = data.apply(word_corrections)
  data = data.apply(remove_punctions_betterway)
  data = data.apply(remove_stop_words)
  data = data.apply(spacy_tokenisation)
  data = data.apply(stemming_data)
  return data


def text_preprocess(data): 
  
  data = remove_html_tags(data)  
  data = remove_urls(data)  
  data = data.lower()
  data = word_corrections(data)
  data = remove_punctions_betterway(data)
  data = remove_stop_words(data)
  data = spacy_tokenisation(data)
  data = stemming_data(data)
  
  return data



def text_pipeline(data,tokenizer):
  txt = text_preprocess(data)
  seq = tokenizer.texts_to_sequences([txt])
  X   = pad_sequences(seq, maxlen=1632)
  return np.array(X)



def model_predict_nn(seq,model):
  results= np.array(['business', 'entertainment', 'politics', 'sport', 'technology'])
  y_pred = model.predict(seq)
  y_pred = np.argmax(y_pred)
  print(y_pred)
  print('The entered article is classified to -- ',results[y_pred]) 
  return results[y_pred]

def model_predict_rf(seq,model):
  results= np.array(['business', 'entertainment', 'politics', 'sport', 'tech'])
  y_pred = model.predict(seq)
  print(y_pred)
  print('The entered article is classified to -- ',results[y_pred]) 
  return results[y_pred]

def tokenized_text(data,tokenizer):
  tokenized_text = []
  for caption in data:
    seq = tokenizer.texts_to_sequences([caption])
    tokenized_text.append(seq)
  return tokenized_text



def max_len_captions(data):
  return max(len(caption.split()) for caption in data)



def create_seq(data,max_len):
  X = []
  for sen in data:
    in_seq = pad_sequences(sen, maxlen=max_len)[0]
    X.append(in_seq)

  return X