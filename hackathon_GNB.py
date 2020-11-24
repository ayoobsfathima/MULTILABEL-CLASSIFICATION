# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 19:47:41 2020

@author: Sheeja Ayoob
"""
#import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from skmultilearn.adapt import MLkNN
from sklearn.metrics import hamming_loss, accuracy_score
from sklearn.metrics import f1_score
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB

#import data
df=pd.read_csv(r"C:\Users\Sheeja Ayoob\Desktop\Train.csv")
df_test=pd.read_csv(r"C:\Users\Sheeja Ayoob\Desktop\Test.csv")
ss=pd.read_csv(r"C:\Users\Sheeja Ayoob\Desktop\SampleSubmission.csv")

ID_COL = 'id'

TARGET_COLS = ['Analysis of PDEs', 'Applications',
               'Artificial Intelligence', 'Astrophysics of Galaxies',
               'Computation and Language', 'Computer Vision and Pattern Recognition',
               'Cosmology and Nongalactic Astrophysics',
               'Data Structures and Algorithms', 'Differential Geometry',
               'Earth and Planetary Astrophysics', 'Fluid Dynamics',
               'Information Theory', 'Instrumentation and Methods for Astrophysics',
               'Machine Learning', 'Materials Science', 'Methodology', 'Number Theory',
               'Optimization and Control', 'Representation Theory', 'Robotics',
               'Social and Information Networks', 'Statistics Theory',
               'Strongly Correlated Electrons', 'Superconductivity',
               'Systems and Control']

TOPIC_COLS = ['Computer Science', 'Mathematics', 'Physics', 'Statistics']


X = df['ABSTRACT']
y = np.asarray(df[df.columns[6:]])

vetorizar = TfidfVectorizer(max_features=10000)
_ = vetorizar.fit(list(df['ABSTRACT']) + list(df_test['ABSTRACT']))


# splitting the data to training and testing data set 
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=2) 

# transforming the data 
X_train_tfidf = vetorizar.transform(X_train) 
X_val_tfidf = vetorizar.transform(X_val) 


# using Multi-label kNN classifier 
classifier = BinaryRelevance(GaussianNB())
classifier.fit(X_train_tfidf, y_train)

#prediction
predicted = classifier.predict(X_val_tfidf)


print(f1_score(y_val, predicted,average='micro'))

--------test------------------------------------------------------------------------------


X_test1 = df_test["ABSTRACT"]



X_test1_tfidf = vetorizar.transform(X_test1) 
predicts = classifier.predict(X_test1_tfidf)

k=pd.DataFrame(predicts.todense())
ss[TARGET_COLS] = k
ss.to_csv(r"C:\Users\Sheeja Ayoob\Desktop\hacklive_NLP_GNB.csv", index = False)