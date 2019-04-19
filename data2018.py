# In[]:
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
# In[]:
from pandas import datetime


def parser(x):
    return datetime.strptime(x, '%d/%m/%Y %H:%M:%S')


# read file

df = pd.read_excel('/Users/bounouamustapha/Desktop/data/Passage.xlsx', parse_dates=[2, 3], date_parser=parser)

print("Column headings:")
print(df.columns)

# In[]:
# get only the date of arrival

data = df.drop(columns=['NUM_SEJOUR', 'CODE', 'DATE_SORTIE', 'CCMU', 'GEMSA'], axis=1)
# In[]:
data.index = data['DATE_ARRIVEE']
del data['DATE_ARRIVEE']
data['nb'] = 1

# In[]:
data['nb'] = 1
# In[]:
daily = data.resample('D').sum()
hourly = data.resample('H').sum()
# In[]:
import matplotlib.pyplot as plt
import cufflinks as cf
import plotly.plotly as py
import plotly.graph_objs as go
daily.plot(title="Le nombre des arrivés par jour pour l'année 2018")
plt.show()

# In[]:
data['1/1/2018':'1/15/2018'].resample('H').sum().plot(
    title="Le nombre des arrivés par heures pour 15 premier jours de 2018")
plt.show()
# In[]:
daily.resample('M').mean().plot(title="Le nombre des arrivés par mois pour  2018")
plt.show()
# In[]:


daily['1/1/2018':'1/31/2018'].resample('D').sum().plot(title="Le nombre des arrivés par jour pour janvier 2018")
plt.show()


# In[]:Par jour de semaine
def getPerDay(x, y):
    data_week = daily[x:y]
    data_week['day_of_week'] = data_week.index.weekday_name
    z = 0
    for i in range(0, len(data_week) - len(data_week) % 7):
        if i % 7 == 0:
            z += 1
            dd = data_week[i:i + 6]
            dd = dd.groupby('day_of_week')['nb'].mean()
            if i==0:
                 ax =dd.plot(title='Les arrivés par les jours de semaine pour les ' + str(z) + ' semaines de janvier')
            else :
                dd.plot(ax=ax,title='Les arrivés par les jours de semaine pour les ' + str(z) + ' semaines de janvier')
                ax.legend(range(1,z+1))
    plt.show()
    return


getPerDay('3/1/2018', '3/31/2018')


# In[]:Par jour de semaine heure
from datetime import timedelta
import matplotlib.pyplot as plt
from pandas import DataFrame


def getPerHourWeek(x,y):
    data_hour_week = hourly[x:y]
    data_hour_week['day_of_week'] = data_hour_week.index.weekday_name
    start = data_hour_week.index[0]

    for i in range(0, 7):

            end = start + timedelta(days=1)
            f = data_hour_week[start:end]
            f['nb'].index=range(0,25)
            dmd = DataFrame(index=f.index.hour,data=f['nb'])
            dmd = dmd.head(24)
            start = end
            if i==0:
                ax = dmd.plot(title='Les journées de la semaine 1 au 7 Février 2018 ')
            else:
                dmd.plot(ax=ax)
                ax.legend(['1', '2', '3', '4', '5', '6', '7'])
    return


getPerHourWeek('1/2/2018', '7/2/2018')

plt.show()


# In[]:











# In[]:
def parser(x):
    return datetime.strptime(x, '%d/%m/%y')


arrivals = pd.read_csv('/Users/bounouamustapha/Desktop/data.csv', parse_dates=[0], date_parser=parser)
# In[]:
arrivals.head()
# In[]:
data_2 = arrivals
data_2.index = data_2['ladate']
del data_2['ladate']

##2016


# In[]:
import plotly
from plotly.graph_objs import Scatter, Layout
import plotly.plotly as py
import plotly.graph_objs as go
import pandas as pd
import numpy as np

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

lines=[]


d2015 =data_2['1/1/2015':'12/31/2015'].resample('M').mean()


line = go.Scatter(
    x=months,
    y=d2015,
    mode='lines',
    name='2015'
)
lines.append(line)

d2016 =data_2['1/1/2016':'12/31/2016'].resample('M').mean()


line = go.Scatter(
    x=months,
    y=d2016,
    mode='lines',
    name='2016'
)
lines.append(line)

d2017=data_2['1/1/2017':'12/31/2017'].resample('M').mean()
line = go.Scatter(
    x=months,
    y=d2017,
    mode='lines',
    name='2017'
)
lines.append(line)
d2018=daily.resample('M').mean()
line = go.Scatter(
    x=months,
    y=d2018,
    mode='lines',
    name='2018'
)
lines.append(line)

py.plot(lines)




# In[]:
data_2['1/1/2016':'1/31/2016'].resample('D').mean().plot(title="Le nombre des arrivés par jour pour 2016")

plt.show()

##2017

# In[]:
import matplotlib.pyplot as plt

data_2['1/1/2017':'12/31/2017'].resample('M').mean().plot(title="Le nombre des arrivés par mois pour 2017")
plt.show()
# In[]:
data_2['1/1/2017':'1/31/2017'].resample('D').mean().plot(title="Le nombre des arrivés par jour pour janvier 2017")
plt.show()

############################



# In[] decomposition
from matplotlib import pyplot
from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(hourly['1/1/2018':'1/31/2018'], model='additive')
result.plot()
pyplot.show()
#
# In[] decomposition daily
from matplotlib import pyplot
from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(daily['1/1/2018':'1/31/2018'], model='additive')
result.plot()
pyplot.show()

# In[]:Autocorrelation
#For example, the drug sales time series is a monthly series with patterns repeating every year. So, you can see spikes at 12th, 24th, 36th.. lines.
from pandas.plotting import autocorrelation_plot
autocorrelation_plot(hourly['1/1/2018':'1/7/2018'])
plt.show()


# In[]:Autocorrelation
#For example, the drug sales time series is a monthly series with patterns repeating every year. So, you can see spikes at 12th, 24th, 36th.. lines.
from pandas.plotting import autocorrelation_plot
autocorrelation_plot(daily['1/1/2018':'1/7/2018'])
plt.show()


# In[]:stationnarity daily
from statsmodels.tsa.stattools import adfuller
from numpy import log
result = adfuller(daily['nb'])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])

# In[]:stationnarity hourely
from statsmodels.tsa.stattools import adfuller
from numpy import log
result = adfuller(hourly['nb'])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])









