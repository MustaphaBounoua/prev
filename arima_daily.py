import numpy as np


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def mean_absolute_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    x = np.mean(np.abs(y_true - y_pred))
    return x


# In[] code
import pandas as pd
from pandas import datetime
from matplotlib import pyplot as plt
import numpy as np


def parser(x):
    return datetime.strptime(x, '%d/%m/%Y %H:%M:%S')


df = pd.read_excel('/Users/bounouamustapha/Desktop/data/Passage.xlsx', parse_dates=[2, 3], date_parser=parser)
holidays = pd.read_excel('/Users/bounouamustapha/Desktop/data/holidays2018.xlsx')

data = df.drop(columns=['NUM_SEJOUR', 'CODE', 'DATE_SORTIE', 'CCMU', 'GEMSA'], axis=1)
data.index = data['DATE_ARRIVEE']
del data['DATE_ARRIVEE']
data['nb'] = 1
daily = data.resample('D').sum()

import matplotlib.pyplot as plt

daily.plot(title="Le nombre des arrivés par jour pour l'année 2018")
plt.show()




# In[] test
def sarimax(data, testdate, horizon, nbjourtest, seasonal, seasonality,useexog):
    from pyramid.arima import auto_arima
    from datetime import timedelta
    test_date_time = datetime.strptime(testdate, '%d/%m/%Y')
    end_test = test_date_time + timedelta(days=horizon - 1)
    end_train = test_date_time - timedelta(1)
    start_train = test_date_time - timedelta(days=nbjourtest)
    train = data[start_train:end_train]
    test = data[test_date_time:end_test]
    if useexog:
        print('------------ variables exogene --------------------------')
        train_exogene = getexplanatoryvariables(train)
        test_exogene = getexplanatoryvariables(test)


    print('training set :' + str(start_train) + ' au ' + str(end_train))
    print('test set :' + str(test_date_time) + ' au ' + str(end_test))

    if useexog:
        arima_model = auto_arima(train, exogenous=train_exogene, seasonality=False, error_action='ignore',
                             trace=1, stepwise=True)
    else:
        arima_model = auto_arima(train, seasonality=False,
                                 error_action='ignore',
                                 trace=1, stepwise=True)

    if useexog:
        prevision = arima_model.predict(horizon, exogenous=test_exogene)
    else:
        prevision = arima_model.predict(horizon)

    precision = mean_absolute_percentage_error(test, prevision)
    print(arima_model.summary())

    print('-----------------------------------------------------------------------------')
    print('--------Mape : --------' + str(precision) + '--------------------------------------')

    x = daily[start_train:end_test]

    plt.plot(x.index, x)
    plt.plot(test.index, prevision)

    plt.legend(['observation', 'prevision'])
    plt.title('La prevision sur un horizon de  :' + str(horizon))
    plt.show()


# In[]

def getexplanatoryvariables(data):
    days_week = []
    init = [0, 0, 0, 0, 0, 0,0]
    i = 0
    for index, item in data.iterrows():
        day = np.array(init)
        if index.weekday() < 6:
            day[index.weekday()] = 1
        print('tes==='+ str(holidays['holiday'][i]))
        if holidays['holiday'][i]:
            day[6] = 1
        days_week.append(day)
        i += 1
    x = np.transpose(days_week)
    return pd.DataFrame({'lundi': x[0], 'Mardi': x[1], 'Mercredi': x[2], 'Jeudi': x[3], 'Vendredi': x[4], 'Samedi': x[5] ,'holiday':x[6]},index=data.index)






# In[]:

sarimax(daily, '1/7/2018', 15, 180, False, 180,False)

# In[]:

sarimax(daily, '28/1/2018', 4, 20, False, 30,True)

# In[]:
import plotly.plotly as py
import plotly.graph_objs as go
import pandas as pd
import numpy as np

da = daily['1/1/2018':'31/12/2018']

x = getexplanatoryvariables(da)


i = 0
z=s=k=0

for index, row in da.iterrows():
    if (x['holiday'].loc[index]):
        print("ok")
        s +=row['nb']
        i += 1
    z +=row['nb']
    k +=1

print('moyenne en vacance :' + str (s/i))
print('moyenne sans vacance :' + str (z/k))



