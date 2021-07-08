#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd 
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


df= pd.read_csv(r"F:\WORK\Machine Learning\Linear Regression\homeprices.csv")


# In[7]:


df


# In[12]:


plt.xlabel('area')
plt.ylabel('price')
plt.scatter(df.area,df.price,color='red',marker='+')


# In[16]:


new_df=df.drop('price',axis='columns')
new_df


# In[17]:


model= linear_model.LinearRegression()
model.fit(new_df,df.price)


# In[21]:


model.predict([[5000]])


# In[22]:


model.coef_


# In[23]:


model.intercept_


# In[26]:


area=pd.read_csv(r"F:\WORK\Machine Learning\Linear Regression\areas.csv")


# In[28]:


area


# In[31]:


p=model.predict(area)


# In[32]:


area['prices']=p


# In[33]:


area


# In[35]:


area.to_csv("F:\WORK\Machine Learning\Linear Regression\predection.csv")


# In[ ]:




