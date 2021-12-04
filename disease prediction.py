#!/usr/bin/env python
# coding: utf-8

# ### Importing libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ### Importing and Cleaning of the Dataset

# In[2]:


data = pd.read_csv("Training.csv")


# In[3]:


data.head()


# In[4]:


## removing the last column of the dataframe
data.drop(data.columns[-1], axis=1,inplace=True)


# In[5]:


data.head()


# ### Exploring the dataset

# In[6]:


data.shape


# In[7]:


data.info(memory_usage="deep")


# In[8]:


data.describe()


# In[9]:


data.isnull().any()


# - Decreasing memory usage by decreasing the size of different columns with int64

# In[10]:


list_col_int64 = [col for col in data.columns if data[col].dtype == "int64"]
# a list to contain the names of all the columns of the dataset


# In[11]:


for col in list_col_int64 :
    data[col] = pd.to_numeric(data[col], downcast="unsigned")


# In[12]:


data.info(memory_usage="deep")


# **********************************************************

# - Memory usage :
# - Before : 5.3 MB
# - After : 971.3 Kb

# ### Data preprocessing

# - Converting categorical data into numerical data

# ##### Dummy variable method

# In[13]:


prognosis_dummies = pd.get_dummies(data.prognosis, prefix="Disease", drop_first="True")
prognosis_dummies.head()


# In[14]:


data_merged = pd.concat([data, prognosis_dummies], axis=1)
data_merged.head()


# In[15]:


data_merged.shape


# In[16]:


df = data_merged.drop("prognosis", axis=1)
df.shape


# ##### Label encoding method

# In[17]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[18]:


data_le = pd.DataFrame(le.fit_transform(data[["prognosis"]]), columns=["Disease"])


# In[19]:


data_le


# In[20]:


merging_data = pd.concat([data, data_le], axis=1)
merging_data.head(3)


# In[21]:


disease_prognosis = pd.DataFrame(merging_data.prognosis)
disease_prognosis["Disease"] = merging_data.Disease
# disease_prognosis.drop_duplicates()
disease_prognosis.head(3)


# In[22]:


disease_prognosis.shape


# In[23]:


disease_prognosis = disease_prognosis.drop_duplicates().reset_index()
disease_prognosis.drop("index", axis=1, inplace=True)
disease_prognosis.head(3)


# In[24]:


disease_dict = {}


# In[25]:


# final_data = merging_data.drop("prognosis",axis=1)
# you can remove the columns later


# In[26]:


merging_data.head(3)


# In[27]:


X_train = merging_data.drop(["prognosis", "Disease"], axis=1)
y_train = merging_data.Disease


# ### Model creation

# In[28]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()


# In[29]:


lr.fit(X_train,y_train)


# In[30]:


test_data = pd.read_csv("Testing.csv")
test_data.head(3)


# In[31]:


test_data["Disease"] = le.fit_transform(test_data.prognosis)
test_data.head(3)


# In[32]:


X_test = test_data.drop(["Disease", "prognosis"], axis=1)
y_test = test_data.Disease


# In[33]:


X_test.shape


# In[34]:


lr.score(X_test, y_test)


# #### Algorithm : Logistic Regression
# #### Model accuracy : 1.0
