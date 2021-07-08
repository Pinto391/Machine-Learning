#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import linear_model


# In[2]:


df=pd.read_csv(r"F:\WORK\Machine Learning\Linear Regression (1)\canada_per_capita_income.csv")


# In[3]:


df


# In[11]:


plt.xlabel('year')
plt.ylabel('per capita income (US$)')
plt.scatter(df.year,df.per_capita_income,color='orange',marker='+')


# In[14]:


new_df=df.drop('per_capita_income',axis='columns')
new_df


# In[15]:


model=linear_model.LinearRegression()
model.fit(new_df,df.per_capita_income)


# In[18]:


model.predict([[2020]])


# In[ ]:




