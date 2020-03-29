#!/usr/bin/env python
# coding: utf-8

# In[3]:


import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import multinomial
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler
import seaborn as sb


# In[34]:


data =  pd.read_csv('listings.csv')
data = data.dropna()

multinomial_distribution = data[['id','host_id','price','minimum_nights','number_of_reviews','calculated_host_listings_count','availability_365']]
multinomial_distribution.hist()
plt.subplots_adjust(hspace = 1, wspace = 1) 
plt.rcParams["figure.figsize"] = [20,15] #size histogram
plt.show()


# In[35]:


normal_distribution = data[['latitude','longitude','reviews_per_month']]
normal_distribution.hist()
plt.subplots_adjust(hspace = 0.5, wspace = 0.5) 
plt.rcParams["figure.figsize"] = [20,15] #size histogram
plt.show()




# In[ ]:





# In[ ]:





# In[41]:



sb.distplot(normal_distribution['latitude'])
plt.subplots_adjust(hspace = 1, wspace = 1) 
plt.rcParams["figure.figsize"] = [20,15] #size histogram
plt.show()


# In[26]:


sb.distplot(normal_distribution['longitude'])


# In[27]:



sb.distplot(normal_distribution['reviews_per_month'])


# In[33]:


sb.distplot(data[['id']])


# In[36]:


sb.distplot(data[['host_id']])


# In[37]:


sb.distplot(data[['price']])


# In[38]:


sb.distplot(data[['number_of_reviews']])


# In[39]:


sb.distplot(data[['calculated_host_listings_count']])


# In[40]:


sb.distplot(data['availability_365'])


# In[ ]:




