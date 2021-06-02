#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

df=pd.read_csv('C:/Users/Aayushi/Desktop/Data Science/ML Revision(By Afsaan)/creditcard.csv')
df.head()

df.shape

df.Class.value_counts()

X=df.drop('Class',axis=1)
y=df.Class

X.head()

y.head()

X=X.drop('Time',axis=1)

X.head()

from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test=train_test_split(X , y , test_size=0.3 , random_state=42 )

#applying Logistic Regression
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(X_train , y_train)

#Evaluating Performance
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
y_pred=logreg.predict(X_test)
print(confusion_matrix(y_test , y_pred))
print(accuracy_score(y_test , y_pred))
print(classification_report(y_test , y_pred))

#Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV
logreg=LogisticRegression()
grid={'C':10.0**np.arange(-1,6) , 'penalty':['l1','l2']}

from sklearn.model_selection import KFold
cv=KFold(n_splits=6 , shuffle=False)

print(cv)


10.0**np.arange(-1,6)


clf=GridSearchCV(logreg , grid , scoring='accuracy' , n_jobs=-1 , cv=cv )


print(clf)

clf.fit(X_train , y_train)

#Evaluating Performance after hyperparameter tuning
y_pred=clf.predict(X_test)
print(confusion_matrix(y_test , y_pred))
print(accuracy_score(y_test , y_pred))
print(classification_report( y_test , y_pred))

from sklearn.ensemble import RandomForestClassifier
ran_forest=RandomForestClassifier()

ran_forest.fit(X_train , y_train)

y_pred=ran_forest.predict(X_test)
print(confusion_matrix(y_test , y_pred))
print(accuracy_score(y_test , y_pred))
print(classification_report( y_test , y_pred))

#Logistic Matrix
#tp: 85293  increase this value
#tn: 85     increase this value
#fp: 14     decrease this value
#fn: 51     decrease this value

#random forest matrix
#tp: 85301    --->increased
#tn: 110      --->increased   
#fp: 6      --->decreased
#fn: 26    --->decreased


#undersampling
from imblearn.under_sampling import NearMiss
from collections import Counter
under_samp=NearMiss(0.85)
X_train_under_samp , y_train_under_samp=under_samp.fit_resample(X_train , y_train)
print('The Classes before undersampling {}'.format(Counter(y_train)))
print('The Classes after undersampling {}'.format(Counter(y_train_under_samp)))

ran_forest.fit(X_train_under_samp , y_train_under_samp)

y_pred=ran_forest.predict(X_test)
print(confusion_matrix(y_test , y_pred))
print(accuracy_score(y_test , y_pred))
print(classification_report(y_test , y_pred))


#random forest matrix after undersampling
#tp:  2932   
#tn:  136       
#fp:  82375     
#fn:   0

#Oversampling
from imblearn.over_sampling import RandomOverSampler , SMOTE 
over_samp=RandomOverSampler(0.85)
X_train_over_samp , y_train_over_samp = over_samp.fit_resample(X_train , y_train)
print('The Classes before oversampling {}'.format(Counter(y_train)))
print('The Classes after after oversampling {}'.format(Counter(y_train_over_samp)))

ran_forest.fit(X_train_over_samp , y_train_over_samp)

y_pred=ran_forest.predict(X_test)
print(confusion_matrix(y_test , y_pred))
print(accuracy_score(y_test , y_pred))
print(classification_report(y_test , y_pred))

#random forest after oversampling
# tp:85300
# tn:112
# fp:7
# fn:24

#random forest matrix
#tp: 85301    
#tn: 110      
#fp: 6     
#fn: 26   

#Oversampling
smote=SMOTE(0.85)
X_train_smote , y_train_smote = smote.fit_resample(X_train , y_train)
print('The Classes before SMOTE {}'.format(Counter(y_train)))
print('The Classes after after SMOTE {}'.format(Counter(y_train_smote)))

ran_forest.fit(X_train_smote , y_train_smote )

y_pred=ran_forest.predict(X_test)
print(confusion_matrix(y_test , y_pred))
print(accuracy_score(y_test , y_pred))
print(classification_report(y_test , y_pred))

#random forest after SMOTE
# tp:85284
# tn:119
# fp:23
# fn:17


#random forest matrix
#tp: 85301    
#tn: 110      
#fp: 6     
#fn: 26   

#random forest after oversampling
# tp:85300
# tn:112
# fp:7
# fn:24


