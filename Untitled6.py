#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from datetime import datetime


# In[2]:


data = pd.read_csv('try.csv')
data.head()


# In[3]:


data.info()


# In[5]:


data['date'] = pd.to_datetime(data['date'])
data['date']


# In[6]:


day = data['date'].dt.day
day


# In[7]:


month = data['date'].dt.month
month


# In[8]:


year = data['date'].dt.year
year


# In[18]:


df1 = pd.DataFrame({'day':day,'month':month,'year':year}) 
df1


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[26]:


df2= pd.read_csv('try2.csv')


# In[28]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df1, df2, test_size=0.2, random_state=0)


# In[29]:


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[58]:


from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(X_train, y_train)
y_pred1 = regressor.predict(X_test)
y_pred1


# In[57]:


from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred1))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred1))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred1)))


# In[39]:


data2 = pd.read_csv('input.csv')
data2.head()
data2['Date']


# In[40]:


data2['Date'] = pd.to_datetime(data2['Date'])
data2['Date']


# In[46]:


day = data2['Date'].dt.day
day


# In[47]:


month = data2['Date'].dt.month
month


# In[48]:


year = data2['Date'].dt.year
year


# In[49]:


data_input = pd.DataFrame({'day':day,'month':month,'year':year}) 
data_input


# In[51]:


y_pred = regressor.predict(data_input)


# In[59]:


y_pred


# In[ ]:





# In[ ]:




