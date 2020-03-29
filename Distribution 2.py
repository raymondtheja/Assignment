#!/usr/bin/env python
# coding: utf-8

# In[50]:


import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import multinomial
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler


# In[62]:


data =  pd.read_csv('listings.csv')
data = data.dropna()

multinomial_distribution = data[['id','host_id','price','minimum_nights','number_of_reviews','calculated_host_listings_count','availability_365']]
multinomial_distribution.hist()
plt.subplots_adjust(hspace = 1, wspace = 1) 
plt.rcParams["figure.figsize"] = [20,15] #size histogram
plt.show()

multinomial_distribution = MinMaxScaler().fit_transform(multinomial_distribution)
for i in range(len(multinomial_distribution)) : 
    k = 1000
# define the distribution
    dist = multinomial(k, multinomial_distribution[i],7)
    print(dist)
# define a specific number of outcomes from 100 trials
    cases = [33, 33, 34]
# calculate the probability for the case
    pr = dist.pmf(multinomial_distribution[i])
# print as a percentage
    print(' i = %d Case=%s, Probability: %.3f%%' % (i,cases, pr*100))


# In[58]:


normal_distribution = data[['latitude','longitude','reviews_per_month']]
normal_distribution.hist()
plt.subplots_adjust(hspace = 0.5, wspace = 0.5) 
plt.rcParams["figure.figsize"] = [20,15] #size histogram
plt.show()

mu_latitude = 1.314192
sigma_latitude =  0.000935
# create distribution
dist = norm(mu_latitude, sigma_latitude)
# plot pdf
values = [value for value in range(0,3)]
probabilities = [dist.pdf(value) for value in values]
plt.plot(values, probabilities)
plt.show()



# In[59]:


mu_longitude = 103.848787
sigma_longitude =  0.001907
# create distribution
dist = norm(mu_longitude, sigma_longitude)
# plot pdf
values = [value for value in range(0,10000)]
probabilities = [dist.pdf(value) for value in values]
plt.plot(values, probabilities)
plt.show()


# In[60]:



mu_review = 1.043669
sigma_review =  1.653413
# create distribution
dist = norm(mu_review, sigma_review)
# plot pdf
values = [value for value in range(0,10000)]
probabilities = [dist.pdf(value) for value in values]
plt.plot(values, probabilities)
plt.show()


# In[ ]:




