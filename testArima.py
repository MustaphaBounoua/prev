# In[]:
import pandas as pd
from pandas import datetime

# In[]:
def parser(x):
    return datetime.strptime(x, '%d/%m/%y')


arrivals = pd.read_csv('/Users/bounouamustapha/Desktop/data.csv', parse_dates=[0], date_parser=parser)
arrivals = arrivals.loc['2017-01-01':]
# In[]:
arrivals.head()
# In[]:
data=arrivals
data.index = data['ladate']
del data['ladate']
# In[]:
import matplotlib.pyplot as plt
data['1/1/2017':'12/31/2017'].resample('M').mean().plot()
plt.show()

# In[]:
import matplotlib.pyplot as plt
import plotly.plotly as ply
import cufflinks as cf
arrivals.plot(title="Patient arrival Jan 2015--mars 2018")
plt.show()
# In[]:
from plotly.plotly import plot_mpl
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(arrivals, model='multiplicative')
fig = result.plot()
plt.show()
# In[]:
from plotly.plotly import plot_mpl
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(arrivals, model='additive')
fig = result.plot()
plt.show()
# In[]:
from pyramid.arima import auto_arima
train = arrivals.loc['2018-01-01':'2018-03-18']
test=arrivals.loc['2018-03-19':]
stepwise_model = auto_arima(train, start_p=1, start_q=1,
                           max_p=3, max_q=3,m=30,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',
                           suppress_warnings=True,
                           stepwise=True)

print(stepwise_model.aic())
# In[]:

stepwise_model.fit(train)
# In[]:
future_forecast = stepwise_model.predict(n_periods=7)
print(future_forecast)

# In[]:
import matplotlib.pyplot as plt
future_forecast = pd.DataFrame(future_forecast,index = test.index,columns=['Prediction'])
p=pd.concat([test,future_forecast],axis=1)
p.plot()
plt.show()