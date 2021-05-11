#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df=pd.read_csv('C:/Users/Aayushi/Desktop/Data Science/ML Revision(By Afsaan)/creditcard.csv')
df.head()


# In[3]:


df.shape


# In[4]:


df.Class.value_counts()


# In[5]:


X=df.drop('Class',axis=1)
y=df.Class


# In[6]:


X.head()


# In[7]:


y.head()


# In[8]:


X=X.drop('Time',axis=1)


# In[9]:


X.head()


# In[11]:


from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test=train_test_split(X , y , test_size=0.3 , random_state=42 )


# In[13]:


#applying Logistic Regression
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(X_train , y_train)


# In[18]:


#Evaluating Performance
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
y_pred=logreg.predict(X_test)
print(confusion_matrix(y_test , y_pred))
print(accuracy_score(y_test , y_pred))
print(classification_report(y_test , y_pred))


# In[28]:


#Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV
logreg=LogisticRegression()
grid={'C':10.0**np.arange(-1,6) , 'penalty':['l1','l2']}


# In[29]:


from sklearn.model_selection import KFold
cv=KFold(n_splits=6 , shuffle=False)


# In[30]:


print(cv)


# In[32]:


10.0**np.arange(-1,6)


# In[33]:


clf=GridSearchCV(logreg , grid , scoring='accuracy' , n_jobs=-1 , cv=cv )


# In[34]:


print(clf)


# In[37]:


clf.fit(X_train , y_train)


# In[38]:


#Evaluating Performance after hyperparameter tuning
y_pred=clf.predict(X_test)
print(confusion_matrix(y_test , y_pred))
print(accuracy_score(y_test , y_pred))
print(classification_report( y_test , y_pred))


# In[39]:


from sklearn.ensemble import RandomForestClassifier
ran_forest=RandomForestClassifier()


# In[40]:


ran_forest.fit(X_train , y_train)


# In[41]:


y_pred=ran_forest.predict(X_test)
print(confusion_matrix(y_test , y_pred))
print(accuracy_score(y_test , y_pred))
print(classification_report( y_test , y_pred))


# In[ ]:


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


# In[48]:


#undersampling
from imblearn.under_sampling import NearMiss
from collections import Counter
under_samp=NearMiss(0.85)
X_train_under_samp , y_train_under_samp=under_samp.fit_resample(X_train , y_train)
print('The Classes before undersampling {}'.format(Counter(y_train)))
print('The Classes after undersampling {}'.format(Counter(y_train_under_samp)))


# In[49]:


ran_forest.fit(X_train_under_samp , y_train_under_samp)


# In[50]:


y_pred=ran_forest.predict(X_test)
print(confusion_matrix(y_test , y_pred))
print(accuracy_score(y_test , y_pred))
print(classification_report(y_test , y_pred))


# In[ ]:


#random forest matrix after undersampling
#tp:  2932   
#tn:  136       
#fp:  82375     
#fn:   0


# In[52]:


#Oversampling
from imblearn.over_sampling import RandomOverSampler , SMOTE 
over_samp=RandomOverSampler(0.85)
X_train_over_samp , y_train_over_samp = over_samp.fit_resample(X_train , y_train)
print('The Classes before oversampling {}'.format(Counter(y_train)))
print('The Classes after after oversampling {}'.format(Counter(y_train_over_samp)))


# In[53]:


ran_forest.fit(X_train_over_samp , y_train_over_samp)


# In[54]:


y_pred=ran_forest.predict(X_test)
print(confusion_matrix(y_test , y_pred))
print(accuracy_score(y_test , y_pred))
print(classification_report(y_test , y_pred))


# In[ ]:


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


# In[56]:


#Oversampling
smote=SMOTE(0.85)
X_train_smote , y_train_smote = smote.fit_resample(X_train , y_train)
print('The Classes before SMOTE {}'.format(Counter(y_train)))
print('The Classes after after SMOTE {}'.format(Counter(y_train_smote)))


# In[57]:


ran_forest.fit(X_train_smote , y_train_smote )


# In[58]:


y_pred=ran_forest.predict(X_test)
print(confusion_matrix(y_test , y_pred))
print(accuracy_score(y_test , y_pred))
print(classification_report(y_test , y_pred))


# In[ ]:


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


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




