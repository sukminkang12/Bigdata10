#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns


# In[4]:


train = pd.read_csv('./data_set.csv',
                    index_col=0,)
test = pd.read_csv('./data_set_test.csv',
                   index_col=0,)


# In[5]:


train.head()


# In[6]:


test.head()


# In[7]:


train.info()


# In[8]:


test.info()


# In[9]:


from sklearn.model_selection import train_test_split


# In[10]:


train.columns


# In[11]:


X = train[['child', 'elementary_num', 'space_trade', 'marriage',
       'silvertown_1000_aged', 'single_house_hold', 'foreigner',
       'child_education_fa', 'land_value_change', 'num_of_birth',
       'student_per_class', 'single_elderly_per', 'num_of_kindergarten',
       'cultural_fa', 'private_institute', 'num_of_house', 'num_of_elementary',
       'mean_age']]
Y = train['index']


# In[12]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)


# In[13]:


from sklearn.ensemble import RandomForestRegressor
clf = RandomForestRegressor(n_estimators = 350,random_state= 42)
clf.fit(X_train, y_train)


# In[14]:


clf.score(X_train, y_train), clf.score(X_test, y_test)


# In[15]:


pred = clf.predict(X_test)


# In[16]:


from sklearn.metrics import mean_squared_error
from sklearn import metrics
print('MSE:', metrics.mean_squared_error(y_test, pred))


# In[17]:


import numpy as np
plt.scatter(y_test, pred)
plt.plot(np.arange(20),color='r')


# In[18]:


pre = pd.read_csv('./pre_test.csv',
                    index_col=0,encoding = 'cp949')


# In[19]:


pre.head()


# In[20]:


pre.info()


# In[21]:


pre.columns


# In[25]:


X = pre[['유치원 원아수', '초등학교 학생수', '토지거래면적', '혼인건수', '노인 천명당 노인여가복지시설수', '1인가구비율',
       '등록외국인 현황', '유아 천명당 보육시설수', '지가변동률', '출생아수', '학급당 학생수', '독거노인가구비율',
       '유치원수', '인구 십만명당 문화기반시설수', '인구 천명당 사설학원수', '주택수', '초등학교수', '평균연령']]


# In[26]:


pred = clf.predict(X)


# In[27]:


pred[0]
i = 0
stand = 0


# In[30]:


for i in range(0, len(pred)):
    if(i%19==0):
        print('\n')
        stand = pred[i]
    else:
        print(stand-pred[i])
    


# In[16]:


from sklearn.tree import DecisionTreeRegressor
clf2 = DecisionTreeRegressor()
clf2.fit(X_train, y_train)


# In[17]:


pred = clf2.predict(X_test)


# In[18]:


print('MSE:', metrics.mean_squared_error(y_test, pred))


# In[31]:


plt.scatter(y_test,pred)
plt.plot(np.arange(20),color='r')


# In[32]:


from sklearn.linear_model import Ridge
ridgeReg = Ridge(alpha = 0.15,normalize =True)
ridgeReg.fit(X_train,y_train)

pred = ridgeReg.predict(X_test)

print('MSE:',metrics.mean_squared_error(y_test,pred))


# In[33]:


plt.scatter(y_test,pred)
plt.plot(np.arange(20),color='r')


# In[38]:


from sklearn.linear_model import Lasso
lassoReg = Lasso(alpha =0.005,normalize = True)
lassoReg.fit(X_train,y_train)

pred = lassoReg.predict(X_test)
print('MSE:',metrics.mean_squared_error(y_test,pred))


# In[39]:


plt.scatter(y_test,pred)
plt.plot(np.arange(20),color='r')


# In[40]:


from sklearn.linear_model import ElasticNet
Ela = ElasticNet(alpha = 0.005,l1_ratio = 0.5,normalize = True)
Ela.fit(X_train,y_train)

pred= Ela.predict(X_test)
print('MSE:',metrics.mean_squared_error(y_test,pred))


# In[41]:


plt.scatter(y_test,pred)
plt.plot(np.arange(20),color='r')


# In[ ]:




