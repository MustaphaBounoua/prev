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

def sarima_prim(data,p,d,q,P,D,Q,s):
    import statsmodels.api as sm
    model = sm.tsa.statespace.SARIMAX(data, order = (p, d, q),seasonal_order = (P, D, Q, s)).fit(disp=-1)
    return model



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


# In[] test
def tsplot(y, lags=None, figsize=(12, 7), style='bmh'):
    import statsmodels.api as sm
    import statsmodels.tsa.api as smt
    from matplotlib import pyplot as plt
    """
        Plot time series, its ACF and PACF, calculate Dickey–Fuller test
        y - timeseries
        lags - how many lags to include in ACF, PACF calculation
    """
    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    with plt.style.context(style):
        fig = plt.figure(figsize=figsize)
        layout = (2, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))

        y.plot(ax=ts_ax)
        p_value = sm.tsa.stattools.adfuller(y)[1]
        ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value))
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
        plt.tight_layout()
        plt.show()


# In[] test
model=sarima_prim(daily['nb'],3,0,2,0,1,1,7)
# In[] test
tsplot(model.resid,20)


# In[] test
def testarima(data, testdate, horizon, nbjourtest, p,d,q,P,D,Q,s):
    from datetime import timedelta
    from pandas import datetime
    import matplotlib.pyplot as plt
    print(str(nbjourtest))
    print(str(horizon))
    test_date_time = datetime.strptime(testdate, '%d/%m/%Y')
    end_test = test_date_time + timedelta(days=horizon - 1)
    end_train = test_date_time - timedelta(1)
    start_train = test_date_time - timedelta(days=nbjourtest)
    train = data[start_train:end_train]
    test = data[test_date_time:end_test]
    arima_model = sarima(train,p,d,q,P,D,Q,s)
    prevision = arima_model.predict(horizon)
    precision = mean_absolute_percentage_error(test, prevision)
    print(arima_model.summary())
    print('-----------------------------------------------------------------------------')
    print('-------- Mape : --------' + str(precision) + '--------------------------------------')
    plt.plot(test.index, test)
    print('-------- test : --------' + str(len(test)))
    print('-------- horizon : --------' + str(horizon))
    print('-------- prevision : --------' + str(len(prevision)))
    plt.plot(np.arange(358), prevision)
    plt.legend(['observation', 'prevision'])
    plt.title('La prevision sur un horizon de  :' + str(horizon))
    plt.show()


# In[
def optimizeSARIMA(data,parameters_list, d, D, s):
    import statsmodels.api as sm

    results = []
    best_aic = float("inf")
    i=0
    for param in parameters_list:
        print("----------------------------------------------------")
        print("--"+ str(i + 1)+"/"+str(len(parameters_list)))
        print("ARIMA "+ "("+str(param[0])+","+str(d)+","+str(param[1])+") ("+str(param[2])+","+str(D)+","+str(param[3])+")"+str(s))
        try:
            model = sm.tsa.statespace.SARIMAX(data, order=(param[0], d, param[1]),seasonal_order=(param[2], D, param[3], s)).fit(disp=-1)
            print("fitting")
            print("----------------------------------------------------")
        except :
            print("Infini")
            print("----------------------------------------------------")
            continue
        aic = model.aic
        # saving best model, AIC and parameters
        if aic < best_aic:
            best_model = model
            best_aic = aic
            best_param = param
        results.append([param, model.aic])

    result_table = pd.DataFrame(results)
    result_table.columns = ['parameters', 'aic']
    # sorting in ascending order, the lower AIC is - the better
    result_table = result_table.sort_values(by='aic', ascending=True).reset_index(drop=True)
    return result_table



    # In[
ps = range(0, 4)
d = 0
qs = range(0, 4)
Ps = range(0, 3)
D = 1
Qs = range(0, 3)

from itertools import product
parameters = product(ps, qs, Ps, Qs)

parameters_list = list(parameters)

aa=daily['1/7/2017':'30/6/2018']

optimizeSARIMA(aa,parameters_list,d,D,7)

# In[
aa=daily['1/7/2017':'31/12/2018']
testarima(aa,'1/7/2018', 100,365,2,0,2,0,1,1,7)
