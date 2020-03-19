#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd

data = pd.read_csv('listings.csv')
discrete_data = data[['id', 'host_id', 'price', 'minimum_nights', 'number_of_reviews', 'calculated_host_listings_count', 'availability_365']]
continous_data = data[['latitude','longitude','reviews_per_month']]
print(discrete_data.head())
print(continous_data.head())


# In[6]:


discrete_data.mean()


# In[7]:


continous_data.mean()


# In[9]:


discrete_data.var()


# In[10]:


continous_data.var()


# In[ ]:




