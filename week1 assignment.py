#!/usr/bin/env python
# coding: utf-8

# # WEEK 1 Assignment

# In[5]:


import pandas as pd


listingData = pd.read_csv("listings.csv")


# In[13]:


# Find Maximum Value
maxValue = listingData.max()
print("\t  - Maximum Value All Data - \n")
print(maxValue)
print()


# In[14]:


# Find Minimum Value
minValue = listingData.min()
print("\t  - Minimum Value All Data - \n")
print(minValue)
print()


# In[19]:


# Eliminate Null Value
listingDataDrop = listingData.dropna()
print(listingDataDrop.isna().values.any())
print(len(listingDataDrop))
print(len(listingData))


# In[21]:


# Replace Null Value to 0
listingDataFill = listingData.fillna(0)
print(listingDataFill.isna().values.any())
print(len(listingData))
print(len(listingDataFill))


# In[ ]:




