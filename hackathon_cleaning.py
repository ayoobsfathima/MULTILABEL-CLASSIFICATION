# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 20:03:15 2020

@author: Sheeja Ayoob
"""
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")
def cleanHtml(sentence):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', str(sentence))
    return cleantext
def cleanPunc(sentence): #function to clean the word of any punctuation or special characters
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n"," ")
    return cleaned
def keepAlpha(sentence):
    alpha_sent = ""
    for word in sentence.split():
        alpha_word = re.sub('[^a-z A-Z]+', ' ', word)
        alpha_sent += alpha_word
        alpha_sent += " "
    alpha_sent = alpha_sent.strip()
    return alpha_sent
df['ABSTRACT'] = df['ABSTRACT'].str.lower()
df['ABSTRACT'] = df['ABSTRACT'].apply(cleanHtml)
df['ABSTRACT'] = df['ABSTRACT'].apply(cleanPunc)
df['ABSTRACT'] = df['ABSTRACT'].apply(keepAlpha)


df_test['ABSTRACT'] = df_test['ABSTRACT'].str.lower()
df_test['ABSTRACT'] = df_test['ABSTRACT'].apply(cleanHtml)
df_test['ABSTRACT'] = df_test['ABSTRACT'].apply(cleanPunc)
df_test['ABSTRACT'] = df_test['ABSTRACT'].apply(keepAlpha)



stop_words = set(stopwords.words('english'))
stop_words.update(['zero','one','two','three','four','five','six','seven','eight','nine','ten','may','also','across','among','beside','however','yet','within'])
re_stop_words = re.compile(r"\b(" + "|".join(stop_words) + ")\\W", re.I)
def removeStopWords(sentence):
    global re_stop_words
    return re_stop_words.sub(" ", sentence)
df['ABSTRACT'] = df['ABSTRACT'].apply(removeStopWords)

df_test['ABSTRACT'] = df_test['ABSTRACT'].apply(removeStopWords)

stemmer = SnowballStemmer("english")
def stemming(sentence):
    stemSentence = ""
    for word in sentence.split():
        stem = stemmer.stem(word)
        stemSentence += stem
        stemSentence += " "
    stemSentence = stemSentence.strip()
    return stemSentence
df['ABSTRACT'] = df['ABSTRACT'].apply(stemming)

df_test['ABSTRACT'] = df_test['ABSTRACT'].apply(stemming)