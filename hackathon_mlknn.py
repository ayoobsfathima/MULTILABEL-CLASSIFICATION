# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 14:58:15 2020

@author: Sheeja Ayoob
"""

#import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from skmultilearn.adapt import MLkNN
from sklearn.metrics import f1_score

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



X =(df[df.columns[1:6]])
y = np.asarray(df[df.columns[6:]])
X_test1 = (df_test[df_test.columns[1:6]])



vetorizar = TfidfVectorizer(max_features=10000)
_ = vetorizar.fit(list(df['ABSTRACT']) + list(df_test['ABSTRACT']))


# splitting the data to training and testing data set 
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.30,random_state=2) 


# transforming the data 
X_train_tfidf = vetorizar.transform(X_train) 
X_val_tfidf = vetorizar.transform(X_val) 
X_test1_tfidf = vetorizar.transform(X_test1) 


# using Multi-label kNN classifier 
mlknn_classifier = MLkNN() 
mlknn_classifier.fit(X_train_tfidf, y_train) 

#prediction
predicted = mlknn_classifier.predict(X_val_tfidf)


print(f1_score(y_val, predicted,average='micro'))

--------test------------------------------------------------------------------------------




predicts = mlknn_classifier.predict(X_test1_tfidf)

k=pd.DataFrame(predicts.todense())
ss[TARGET_COLS] = k
ss.to_csv(r"C:\Users\Sheeja Ayoob\Desktop\hacklive_NLP_sub7.csv", index = False)
--------------------------------------------------------------------------------------------

#optimal threshold
def get_best_thresholds(true, preds):
  thresholds = [i/100 for i in range(100)]
  best_thresholds = []
  for idx in range(25):
    f1_scores = [f1_score(true[:, idx], (preds[:, idx] > thresh) * 1) for thresh in thresholds]
    best_thresh = thresholds[np.argmax(f1_scores)]
    best_thresholds.append(best_thresh)
  return best_thresholds

val_preds = mlknn_classifier.predict_proba(X_val_tfidf)
val_preds=val_preds.toarray()

best_thresholds = get_best_thresholds(y_val,val_preds)

for i, thresh in enumerate(best_thresholds):
  val_preds[:, i] = (val_preds[:, i] > thresh) * 1
  
f1_score(y_val, val_preds, average='micro')


preds_test = mlknn_classifier.predict_proba(X_test1_tfidf)

for i, thresh in enumerate(best_thresholds):
  preds_test[:, i] = (preds_test[:, i] > thresh) * 1
  
k=pd.DataFrame(preds_test.todense())
ss[TARGET_COLS] = k
ss.to_csv(r"C:\Users\Sheeja Ayoob\Desktop\hacklive_NLP_sub8.csv", index = False)
--------------------------------------------------------------------------------------------
#combining topics

vetorizar = TfidfVectorizer(max_features=10000)
_ = vetorizar.fit(list(X['ABSTRACT']) + list(X_test1['ABSTRACT']))


# splitting the data to training and testing data set 
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.30, random_state=2) 

# transforming the data 
X_train_tfidf = vetorizar.transform(X_train['ABSTRACT']) 
X_val_tfidf = vetorizar.transform(X_val['ABSTRACT']) 
X_test1_tfidf = vetorizar.transform(X_test1['ABSTRACT']) 

trn2 = np.hstack((X_train_tfidf.toarray(), X_train[TOPIC_COLS]))
val2 = np.hstack((X_val_tfidf.toarray(), X_val[TOPIC_COLS]))
tst2 = np.hstack((X_test1_tfidf.toarray(), df_test[TOPIC_COLS]))

from scipy.sparse import csr_matrix

trn2 = csr_matrix(trn2.astype('int16'))
val2 = csr_matrix(val2.astype('int16'))
tst2 = csr_matrix(tst2.astype('int16'))

# using Multi-label kNN classifier 
mlknn_classifier = MLkNN() 
mlknn_classifier.fit(trn2, y_train) 

#prediction
predicted = mlknn_classifier.predict(val2)


print(f1_score(y_val, predicted,average='micro'))

predicts = mlknn_classifier.predict(tst2)  
k=pd.DataFrame(predicts.todense())
ss[TARGET_COLS] = k
ss.to_csv(r"C:\Users\Sheeja Ayoob\Desktop\hacklive_NLP_sub7.csv", index = False)






val_preds = mlknn_classifier.predict_proba(val2)
val_preds=val_preds.toarray()
best_thresholds = get_best_thresholds(y_val,val_preds)

for i, thresh in enumerate(best_thresholds):
  val_preds[:, i] = (val_preds[:, i] > thresh) * 1

f1_score(y_val, val_preds, average='micro')