#!/usr/bin/env python
# coding: utf-8

# # Time Series Forescasting

# In[3]:
import pandas as pd
from pandas import datetime
import matplotlib.pyplot as plt


# In[2]:


def parser(x):
    return datetime.strptime(x, '%d/%m/%y')


arrivals = pd.read_csv('/Users/bounouamustapha/Desktop/data.csv', index_col=0, parse_dates=[0], date_parser=parser)

# In[1]:


arrivals.head()

# In[17]:


arrivals.plot()
plt.show()
# Stationary means mean, variance and covariance is constant over periods.

# In[23]:


from statsmodels.graphics.tsaplots import plot_acf

plot_acf(arrivals)
plt.show()
# In[ ]:


# In[ ]:


# ### Converting series to stationary

# In[ ]:


# In[18]:

arrivals.head()

# In[24]:


arrivals.shift(1)

# In[20]:


sales_diff = arrivals.diff(periods=1)
# integrated of order 1, denoted by d (for diff), one of the parameter of ARIMA model


# In[22]:


sales_diff = sales_diff[1:]
sales_diff.head()

# In[25]:


plot_acf(sales_diff)
plt.show()
# In[26]:


sales_diff.plot()

# In[70]:


X = arrivals.values
st=732
all=X[732:]
start=len(all)-10
train = all[:(len(all)-10)]  # 27 data as train data
test = all[len(all)-10:]  # 9 data as test data
predictions = []
# In[ ]: all
all.size



# In[63]:

train.size


# In[ ]:
test.size

# In[41]:


from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error

model_ar = AR(train)
model_ar_fit = model_ar.fit()

# In[50]:


predictions = model_ar_fit.predict(start=start, end=start+10)

# In[51]:


test

# In[52]:


plt.plot(test)
plt.plot(predictions, color='red')
plt.show()
# In[49]:


arrivals.plot()

# # ARIMA model

# In[53]:


from statsmodels.tsa.arima_model import ARIMA

# In[102]:


# p,d,q  p = periods taken for autoregressive model
# d -> Integrated order, difference
# q periods in moving average model
model_arima = ARIMA(train, order=(1, 2, 0))
model_arima_fit = model_arima.fit()
print(model_arima_fit.aic)

# In[103]:


predictions = model_arima_fit.forecast(steps=10)[0]
predictions

# In[104]:


plt.plot(test)
plt.plot(predictions, color='red')

# In[97]:


mean_squared_error(test, predictions)

# In[82]:


import itertools

p = d = q = range(0, 5)
pdq = list(itertools.product(p, d, q))
pdq

# In[86]:


import warnings

warnings.filterwarnings('ignore')
for param in pdq:
    try:
        print(param, param)
        model_arima = ARIMA(train, order=param)
        model_arima_fit = model_arima.fit()
        print(param, model_arima_fit.aic)
    except:
        continue


