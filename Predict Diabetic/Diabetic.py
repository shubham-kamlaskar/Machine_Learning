#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import re
import os


# In[2]:


dia = open(r"C:\Users\shubham\Desktop\Python\Kaggle\Predict Diabetic\diabetes.csv")


# In[3]:


diabetic = pd.read_csv(dia)


# In[4]:


diabetic


# In[5]:


diabetic.info()


# In[6]:


diabetic.describe()


# In[7]:


plt.figure(figsize=[10,5])
sns.heatmap(diabetic.corr(),annot=True)


# In[9]:


diabetic.columns


# In[13]:


X = diabetic.drop(['Outcome'],axis=1)


# In[14]:


y = diabetic['Outcome']


# In[16]:


from sklearn.model_selection import train_test_split


# In[18]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)


# In[19]:


from sklearn.preprocessing import StandardScaler


# In[24]:


from sklearn.tree import DecisionTreeClassifier


# In[28]:


clf = DecisionTreeClassifier()


# In[30]:


clf.fit(X_train,y_train)


# In[35]:


sns.regplot(x=diabetic['Insulin'], y=diabetic['Outcome'])


# In[37]:


from sklearn.linear_model import LogisticRegression


# In[38]:


lr = LogisticRegression()


# In[39]:


lr.fit(X_train,y_train)


# In[40]:


pred = lr.predict(X_test)


# In[41]:


pred


# In[46]:


sns.pairplot(diabetic,hue='Outcome')


# In[47]:


sns.distplot(diabetic)


# In[48]:


lr.intercept_


# In[49]:


lr.coef_


# In[50]:


lr.score(X_train,y_train)


# In[51]:


lr.score(X_test,y_test)


# In[ ]:




