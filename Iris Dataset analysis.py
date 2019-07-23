#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import seaborn as sns;


# In[3]:


import matplotlib.pyplot as plt;


# In[5]:


import pandas as pd


# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


dataset = pd.read_csv('iris.csv')


# In[8]:


dataset.head()


# In[9]:


dataset = dataset.drop('Id', axis=1)
dataset.head()


# In[10]:


print(dataset.shape)


# In[12]:


print(dataset.info())


# In[13]:


print(dataset.describe())


# In[14]:


print(dataset.groupby('Species').size())


# In[15]:


dataset.plot(kind='box', sharex=False, sharey=False)


# In[16]:


dataset.hist(edgecolor='black', linewidth=1.2)


# In[17]:


dataset.boxplot(by="Species", figsize=(10,10))


# In[18]:


sns.violinplot(data=dataset, x="Species", y="PetalLengthCm")


# In[19]:


from pandas.plotting import scatter_matrix


# In[20]:


scatter_matrix(dataset,figsize=(10,10))
plt.show()


# In[21]:


sns.pairplot(dataset, hue='Species')


# In[22]:


sns.pairplot(dataset, hue="Species", diag_kind="kde")


# In[23]:


#Importing metrics for evaluation

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[26]:


# Separating the data into dependent and independent variables

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values



# In[28]:


# Spliting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split


# In[29]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[30]:


from sklearn.linear_model import LogisticRegression


# In[31]:


classifier = LogisticRegression()


# In[32]:


classifier.fit(X_train, y_train)


# In[34]:


y_pred = classifier.predict(X_test)


# In[35]:


print(classification_report(y_test, y_pred))


# In[36]:


print(confusion_matrix(y_test, y_pred))


# In[37]:


from sklearn.metrics import accuracy_score


# In[38]:


print('accuracy is', accuracy_score(y_pred, y_test))


# In[ ]:





# In[39]:


from sklearn.naive_bayes import GaussianNB


# In[40]:


classifier = GaussianNB()


# In[41]:


classifier.fit(X_train, y_train)


# In[42]:


y_pred = classifier.predict(X_test)


# In[43]:


print(classification_report(y_test, y_pred))


# In[44]:


from sklearn.svm import SVC


# In[45]:


classifier = SVC()


# In[46]:


classifier.fit(X_train, y_train)


# In[47]:


y_pred = classifier.predict(X_test)


# In[48]:


print(classification_report(y_test, y_pred))


# In[49]:


print(confusion_matrix(y_test, y_pred))


# In[50]:


from sklearn.metrics import accuracy_score


# In[51]:


print('accuracy is', accuracy_score(y_pred, y_test))


# In[ ]:





# In[ ]:





# In[53]:


from sklearn.neighbors import KNeighborsClassifier


# In[56]:


classifier = KNeighborsClassifier(n_neighbors=8)


# In[57]:


classifier.fit(X_train, y_train)


# In[58]:


y_pred = classifier.predict(X_test)


# In[59]:


print(classification_report(y_test, y_pred))


# In[60]:


print(confusion_matrix(y_test, y_pred))


# In[61]:


from sklearn.metrics import accuracy_score


# In[62]:


print('accuracy is', accuracy_score(y_pred, y_test))


# In[63]:


from sklearn.tree import DecisionTreeClassifier


# In[64]:


classifier = DecisionTreeClassifier()


# In[65]:


classifier.fit(X_train, y_train)


# In[66]:


y_pred = classifier.predict(X_test)


# In[67]:


print(classification_report(y_test, y_pred))


# In[68]:


print(confusion_matrix(y_test, y_pred))


# In[69]:


from sklearn.metrics import accuracy_score


# In[70]:


print('accuracy is', accuracy_score(y_pred, y_test))


# In[ ]:




