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


speed = pd.read_csv('Internet Speed 2022.csv')


# In[3]:


speed


# In[4]:


speed.info()


# In[5]:


speed.isnull().sum()


# In[6]:


speed['mobile'].fillna(speed['mobile'].mean(),inplace=True)


# In[7]:


speed.info()


# In[8]:


speed


# In[9]:


plt.scatter(x=speed['country'],y=speed['broadband'])
plt.show()


# In[10]:


plt.scatter(x=speed['country'],y=speed['mobile'])
plt.show()


# In[11]:


speed1 = speed.sort_values('broadband')


# In[12]:


speed2 = speed.sort_values('mobile')


# In[13]:


heatmap = speed.corr()

sns.heatmap(heatmap,annot=True)


# In[14]:


speed1.groupby(['broadband'])


# In[15]:


speed1 = speed1[speed1['broadband'] >150]


# In[16]:


speed1


# In[19]:


speed1 =speed1.sort_values('mobile')

plt.figure(figsize=[10,10])

plt.plot(speed1['country'],speed1['broadband'],color='r')
plt.plot(speed1['country'],speed1['mobile'])

plt.xticks(rotation=45)
plt.show()


# In[26]:


plt.figure(figsize=[10,5])

plt.scatter(speed1['country'],speed1['broadband'],color='r',s = speed1['broadband'],alpha = 0.5)
plt.scatter(speed1['country'],speed1['mobile'],s=speed1['mobile'])

plt.xticks(rotation=45)
plt.show()


# In[49]:


speed3 = speed[speed['broadband'] < 20]
speed3.shape


# In[51]:


plt.figure(figsize=[12,5])

plt.scatter(speed3['country'],speed3['broadband'],color='r',s = speed3['broadband'],alpha = 0.5)
plt.scatter(speed3['country'],speed3['mobile'],s=speed3['mobile'])

plt.xticks(rotation=45)
plt.show()


# In[52]:


speed.describe()


# In[53]:


speed.boxplot()


# In[91]:


a = speed.head(10)
plt.title('Highest boardband speed countries ')

plt.bar(a['country'],a['broadband'])
plt.xticks(rotation=90)
plt.xlabel('Country')
plt.ylabel('Broadband Speed')
plt.show()


# In[88]:


b = speed.tail(10)
plt.title('Lowest boardband speed countries ')
plt.bar(b['country'],b['broadband'],color='r')
plt.xticks(rotation=90)
plt.xlabel('Country')
plt.ylabel('Broadband Speed')
plt.show()


# In[89]:


c = speed.sort_values(by='mobile',ascending=False)

plt.title('Highest Mobile speed countries ')
plt.bar(c['country'].head(15),c['mobile'].head(15),color='y')
plt.xticks(rotation=90)
plt.xlabel('Country')
plt.ylabel('Mobile Speed')
plt.show()


# In[93]:


c = speed.sort_values(by='mobile',ascending=True)

plt.title('Lowest Mobile speed countries ')
plt.bar(c['country'].head(15),c['mobile'].head(15),color='g')
plt.xticks(rotation=90)
plt.xlabel('Country')
plt.ylabel('Mobile Speed')
plt.show()


# In[115]:


a = speed.head(10)
plt.figure(figsize=[10,5])
plt.title('Highest boardband speed countries ')

plt.bar(a['country'],a['broadband'],mouseover=True)
plt.xticks(rotation=90)

plt.bar(a['country'],a['mobile'])
plt.legend(['BroadBand','Mobile'])
plt.xlabel('Country')
plt.ylabel('Broadband & Mobile Speed')
plt.show()


# In[105]:


a = speed.head(10)
plt.title('Highest boardband speed countries ')

plt.bar(a['country'],a['broadband'])
plt.xticks(rotation=90)
plt.plot(a['country'],a['mobile'],color='r')

plt.xlabel('Country')
plt.ylabel('Broadband & Mobile Speed')
plt.show()


# ### India Ranking in world

# In[135]:


speed4 = speed.sort_values(by='broadband',ascending=False )
speed4 =speed4.reset_index()
speed5 = speed.sort_values(by='mobile',ascending=False )
speed5 =speed5.reset_index()
India_b = speed4[speed4['country']=='India']
India_m = speed5[speed5['country']=='India']


# In[137]:


India_b


# India's rank in broadband connection speed is 68th outof 177 countries

# In[138]:


India_m


# India's rank in mobile connection speed is 165th outof 177 countries

# In[153]:


plt.figure(figsize=[12,5])

# plt.bar(speed['country'],speed['broadband'])
plt.bar(speed['country'].head(100),speed['broadband'].head(100),color='r')
plt.plot(India_b['country'],India_b['broadband'],color='b')
plt.show()


# In[ ]:




