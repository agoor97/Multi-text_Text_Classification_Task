#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Most Important
import numpy as np
import pandas as pd

import joblib
import os

## Sklearn
from sklearn import utils

## Preprocessig
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer


# In[2]:


## load the CSV file
df_data = pd.read_csv('Job titles and industries.csv')


# In[3]:


## split the data for train_full & test
split_data_all = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
gen_split_all = split_data_all.split(df_data, df_data['industry'])
## loop over the generator
for train_full_idx, test_full_idx in gen_split_all:
    train_set_full = df_data.iloc[train_full_idx]
    test_set = df_data.iloc[test_full_idx]


## split the data for train and Validation 
split_train = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=123)
gen_split_train = split_train.split(train_set_full, train_set_full['industry'])
## loop over the generator
for train_idx, val_idx in gen_split_train:
    train_set = train_set_full.iloc[train_idx]
    val_set = train_set_full.iloc[val_idx]

## separate Feature and Label
X_train = train_set['job title']
y_train = train_set['industry']

X_val = val_set['job title']
y_val = val_set['industry']

X_test = test_set['job title']
y_test = test_set['industry']

## CountVectorizer for (train, val, test)
vect = TfidfVectorizer()
vect.fit(X_train)
X_train_final = vect.transform(X_train)
X_val_final = vect.transform(X_val)
X_test_final = vect.transform(X_test)


def vectorize_new_instance(X_new):
    ''' this function tries to get the required input to the Model
    Args:
    *****
        (X_new : 1D array of String) ==> the required instance to predict
    '''
    X_new_final = vect.transform(X_new)

    return X_new_final


# In[ ]:





# In[ ]:




