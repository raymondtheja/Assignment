#!/usr/bin/env python
# coding: utf-8

# # Week 2 Assignment

# Regression - price dengan review per month 

# In[30]:


import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# In[31]:


data = pd.read_csv("listings.csv",delimiter = ",")
print(data.isna().values.any())
#isi yang kosong dengan angka 8
datafill = data.fillna(8)
print(datafill.isna().values.any())


# Regression - price

# In[32]:


train,test = train_test_split(datafill,test_size = 0.2)
rlm = linear_model.LinearRegression()
rlm.fit(train[["price"]], train[["reviews_per_month"]])
print("Coefficients : ", rlm.coef_)
print("Intercept : ", rlm.intercept_)


# In[33]:


plt.scatter(train[["price"]], train[["reviews_per_month"]], color = "red")
plt.plot(train[["price"]], rlm.coef_ * train [["price"]] + rlm.intercept_, '-g')
plt.xlabel("Price")
plt.ylabel("Review per Month")
# plt.rcParams["figure.figsize"] = [9,7]
plt.show()


# Predictdata

# In[35]:


prediction = rlm.predict(test[["price"]])
for i in range(len(test)):
  print(test[["price"]].values[i], prediction[i]) #prediksi jika price segini maka reeview per month nya segini

print("MAE : ", mean_absolute_error(test[["reviews_per_month"]], prediction))
print("MSE : ", mean_squared_error(test[["reviews_per_month"]], prediction))
print("R2 : ", r2_score(test[["reviews_per_month"]], prediction))


# # KNN - Classification

# Price, Latitude --> type room

# In[2]:


import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder


# In[3]:


data = pd.read_csv("listings.csv",delimiter = ",")
print(data.isna().values.any())
#isi yang kosong dengan angka 0
datafill = data.fillna(0)
print(datafill.isna().values.any())


# In[4]:


train, test = train_test_split(datafill,test_size = 0.2)


# In[5]:


KNN = KNeighborsClassifier(n_neighbors = 4).fit(train[["price","latitude"]],train["room_type"])


# In[37]:


# predict new data
newdata = KNN.predict([[23,1.45],[18,1.31]])
print(newdata)
print()
csf = KNN.predict(test[["price","latitude"]])
accuracy = accuracy_score(test["room_type"],csf)
array = data[['room_type']]
array = OrdinalEncoder().fit_transform(array)

print("ACC : %.2f"%accuracy)


# In[25]:


n = 30
accuracy = np.zeros((n-1))
for i in range(1, n):    
    KNN = KNeighborsClassifier(n_neighbors = i).fit(train[["price", "latitude"]], train["room_type"])  
    classification = KNN.predict(test[["price", "latitude"]])
    accuracy[i - 1] = accuracy_score(test["room_type"], classification)
    
print("Best  ACC : %.2f" % accuracy.max(), ", with k = ", accuracy.argmax() + 1)


# In[ ]:




