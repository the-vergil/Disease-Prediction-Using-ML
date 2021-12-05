#!/usr/bin/env python
# coding: utf-8

# # Importing libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn


# # Importing and cleaning of dataset

# In[2]:


data = pd.read_csv("Training.csv")
data.head()


# In[3]:


## removing the last column of the dataframe
data.drop(data.columns[-1], axis=1,inplace=True)


# In[4]:


data.head(1)


# In[5]:


data.drop_duplicates(inplace=True)
data = data.reset_index()


# In[6]:


data.shape


# In[7]:


data.head(3)


# # Data exploration

# In[8]:


data.info()


# In[9]:


data.describe()


# In[10]:


data.isnull().any().to_numpy()


# ## Data exploration results : 
# - Shape : (304,133)
# - Dtypes : int64(132), object(1)
# - Memory usage : 318.2 KB
# - No duplicate rows
# - No Nan values

# # Data preprocessing

# ##### Converting categorical data into numerical data
# ###### 1. Method 1 : Dummy variable (OneHotEncoding)
# ###### 2. Method 2 : Label Encoding
# - In this project we're going to use label encoding

# In[11]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[12]:


le_data = pd.DataFrame(le.fit_transform(data[["prognosis"]]), columns=["Disease"])
le_data.tail(5)


# In[13]:


len(le_data)


# In[14]:


len(data)


# In[62]:


merged_data = pd.concat([data,le_data],axis=1)


# In[16]:


merged_data.tail()


# In[63]:


merged_data = merged_data.drop("index", axis=1)


# # Creating csv file for disease and prognosis

# In[24]:


disease_prognosis = pd.DataFrame(merged_data[["prognosis", "Disease"]])
disease_prognosis.head(3)


# In[25]:


disease_prognosis.drop_duplicates(inplace=True)
disease_prognosis = disease_prognosis.reset_index()
len(disease_prognosis)


# In[26]:


disease_prognosis.drop("index", axis=1, inplace=True)


# In[27]:


disease_prognosis.head()


# In[28]:


disease_prognosis.to_csv("Disease Prognosis.csv")


# # Algorithm 1 : Logistic Regression

# In[68]:


X_train = merged_data.drop(["prognosis", "Disease"], axis=1)
y_train = merged_data.Disease


# In[69]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,y_train)


# ### Loading test data

# In[49]:


test_data = pd.read_csv("Testing.csv")
len(test_data)


# In[52]:


test_data.head()


# ### Label encoding the test data

# In[57]:


test_le = pd.DataFrame(le.fit_transform(test_data.prognosis), columns=["Disease"])
test_data = pd.concat([test_data,test_le], axis=1)


# In[59]:


test_data.head(3)


# ### Score of the model

# In[60]:


X_test = test_data.drop(["prognosis", "Disease"], axis=1)
y_test = test_data.Disease


# In[66]:


lr.score(X_test, y_test)


# # Algorithm 2 : SVM

# In[71]:


from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train, y_train)
svm.score(X_test, y_test)


# # Algorithm 3 : RandomForest

# In[72]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train,y_train)
rf.score(X_test, y_test)


# # Algorithm 4 : Decision Tree

# In[73]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)
dt.score(X_test, y_test)


# # Algorithm 5 : Naive Bayes

# In[74]:


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)
gnb.score(X_test, y_test)


# *********************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************

# ### Score for different algorithms are :
# - LogisticRegression : 1.0
# - SVM : 1.0
# - RandomForest : 0.9761904761904762
# - DecisionTree : 0.9761904761904762
# - NaiveBayes : 1.0

# ***************************************************************************************************************************
