# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 13:33:15 2020

@author: Sheeja Ayoob
"""
train = pd.read_csv(r"C:\Users\Sheeja Ayoob\Desktop\Train.csv")
test = pd.read_csv(r"C:\Users\Sheeja Ayoob\Desktop\Test.csv")
ss = pd.read_csv(r"C:\Users\Sheeja Ayoob\Desktop\SampleSubmission.csv")

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

vec = CountVectorizer(max_features=10000)
combined = list(train['ABSTRACT']) + list(test['ABSTRACT'])
vec.fit(combined)

trn, val = train_test_split(train, test_size=0.2, random_state=2)

trn_abs = vec.transform(trn['ABSTRACT'])
val_abs = vec.transform(val['ABSTRACT'])
tst_abs = vec.transform(test['ABSTRACT'])

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
clf = OneVsRestClassifier(LogisticRegression(C = 10, n_jobs=-1))
clf.fit(trn_abs, trn[TARGET_COLS])

val_preds = clf.predict(val_abs)
f1_score(val[TARGET_COLS], val_preds, average='micro')

preds_test = clf.predict(tst_abs)
 ss[TARGET_COLS] = preds_test

 ss.to_csv(r"C:\Users\Sheeja Ayoob\Desktop\hacklive_NLP_sub6.csv", index = False)
----------------------------------------------------------------------------------------
vec = TfidfVectorizer(max_features=10000)
_ = vec.fit(list(train['ABSTRACT']) + list(test['ABSTRACT']))

trn_abs = vec.transform(trn['ABSTRACT'])
val_abs = vec.transform(val['ABSTRACT'])
tst_abs = vec.transform(test['ABSTRACT'])


clf = OneVsRestClassifier(LogisticRegression(C = 10, n_jobs=-1))
_ = clf.fit(trn_abs, trn[TARGET_COLS])

val_preds = clf.predict(val_abs)
f1_score(val[TARGET_COLS], val_preds, average='micro')

def get_best_thresholds(true, preds):
  thresholds = [i/100 for i in range(100)]
  best_thresholds = []
  for idx in range(25):
    f1_scores = [f1_score(true[:, idx], (preds[:, idx] > thresh) * 1) for thresh in thresholds]
    best_thresh = thresholds[np.argmax(f1_scores)]
    best_thresholds.append(best_thresh)
  return best_thresholds

val_preds = clf.predict_proba(val_abs)

best_thresholds = get_best_thresholds(val[TARGET_COLS].values, val_preds)

for i, thresh in enumerate(best_thresholds):
  val_preds[:, i] = (val_preds[:, i] > thresh) * 1
  
f1_score(val[TARGET_COLS], val_preds, average='micro')