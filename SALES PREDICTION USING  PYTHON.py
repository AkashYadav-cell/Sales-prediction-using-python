#!/usr/bin/env python
# coding: utf-8

# In[1]:


#https://www.kaggle.com/code/ashydv/sales-prediction-simple-linear-regression/input


# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# ## importing Dataset

# In[3]:


df=pd.read_csv('advertising.csv')
df.head()


# ## visualization

# In[6]:


df.shape


# In[7]:


df.describe()


# ## Observation
# Average expense spend is highest on tv;
# Average expense spend is lowest on radio;
# max sale is 27 and min is 1.6

# In[11]:


sns.pairplot(df,x_vars=['TV','Radio','Newspaper'],y_vars='Sales',kind='scatter')
plt.show()


# ## Pair plot observation
# when advertiseng cost increases in tv ads ,the sales will increase ass well ,while for newspaper and radio it is bit unpredictable

# In[12]:


df['TV'].plot.hist(bins=10)


# In[15]:


df['Radio'].plot.hist(bins=10,color="green" )


# In[16]:


df['Newspaper'].plot.hist(bins=10,color="purple" )


# ## Histogram Observation
# the majority sales is the result of low advertising cost in newspaper  

# In[17]:


sns.heatmap(df.corr(),annot=True)
plt.show()


# sales is Highly Correlated with TV

# ## Model Training

# In[23]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(df[['TV']],df[['Sales']],test_size=0.3,random_state=0)


# In[24]:


print(x_train)


# In[25]:


print(y_train)


# In[26]:


print(x_test)


# In[28]:


print(y_test)


# In[34]:


from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)


# In[35]:


res=model.predict(x_test)
print(res)


# In[36]:


model.coef_


# In[37]:


model.intercept_


# In[38]:


0.054473199*69.2 + 7.14382225


# In[40]:


plt.plot(res)


# In[41]:


plt.scatter(x_test,y_test)
plt.plot(x_test,0.054473199*x_test + 7.14382225,'r')
plt.show()


# hence the above mention solution is successfully able to predict the sales  using advertising platform datasets 

# In[ ]:




