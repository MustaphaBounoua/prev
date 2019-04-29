# In[
import mypackage as la

data = la.getHourlyData()
train = data['1/7/2017':'30/6/2018']

# In[


model = la.sarima(train.nb,3,0,3,0,1,2,24)

# In[