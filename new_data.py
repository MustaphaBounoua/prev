



# In[] code

from pandas import datetime
from matplotlib import pyplot as plt

import plotly.plotly as py
import plotly.graph_objs as go
import pandas as pd
import numpy as np


def parser(x):
    return datetime.strptime(x, '%d/%m/%Y %H:%M:%S')


df = pd.read_excel('/Users/bounouamustapha/Desktop/data/Passage.xlsx', parse_dates=[2, 3], date_parser=parser)

data = df.drop(columns=['NUM_SEJOUR', 'CODE', 'DATE_SORTIE', 'CCMU', 'GEMSA'], axis=1)
data.index = data['DATE_ARRIVEE']
del data['DATE_ARRIVEE']
data['nb'] = 1
hourly = data.resample('H').sum()
daily = data.resample('D').sum()


df = pd.read_excel('/Users/bounouamustapha/Desktop/data/data201517.xlsx')


from pandas import datetime
data = []

for row in df.itertuples(index=True, name='Pandas'):
    d= getattr(row, "dater")
    d=datetime(year=d.year,month=d.month,day=d.day,hour=getattr(row, "heure"),minute=getattr(row, "minutee"))
    data.append(d)

dd=pd.DataFrame(data,columns=['DATE_ARRIVEE'])
dd.index = dd['DATE_ARRIVEE']
del dd['DATE_ARRIVEE']
dd['nb'] = 1

hourlyold = dd.resample('H').sum()
hourlyold=hourlyold['1/1/2015':'31/12/2017']
dailyold = dd.resample('D').sum()
dailyold=dailyold['1/1/2015':'31/12/2017']


hourlyall = pd.concat([hourly, hourlyold])
hourlyall.sort_index(inplace=True)

# In[
hourlyall.to_csv('/Users/bounouamustapha/Desktop/data/all_data.csv', encoding='utf-8')


# In[] test

def tsplot(y, lags=None, figsize=(12, 7), style='bmh'):
    import statsmodels.api as sm
    import statsmodels.tsa.api as smt
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

# In[

def seasonal_diff(y,lag):
    import statsmodels.tsa.api as smt
    y_diff = y - y.shift(24)
    tsplot(y_diff[24:], lag)
    return y_diff[24:]

# In[
def SARIMA(y,p, d,q,P ,D,Q ,s):
    import statsmodels.api as sm
    model = sm.tsa.statespace.SARIMAX(y, order=(p, d, q),seasonal_order=(P, D, Q, s)).fit(disp=-1)
    return model


# In[
def optimizeSARIMA(y,listparam, d, D, s):
    import statsmodels.api as sm
    from tqdm import tqdm_notebook
    """
        Return dataframe with parameters and corresponding AIC

        parameters_list - list with (p, q, P, Q) tuples
        d - integration order in ARIMA model
        D - seasonal integration order
        s - length of season
    """

    myresults = []
    best_aic = float("inf")

    for param in tqdm_notebook(listparam):
        print("--------*******--------")
        print("param"+"("+str(param[0])+",0,"+str(param[1])+")("+str(param[2])+",1,"+str(param[3])+")")
        print("--------*******--------")
        # we need try-except because on some combinations model fails to converge
        try:
            model = sm.tsa.statespace.SARIMAX(y, order=(param[0], d, param[1]),
                                              seasonal_order=(param[2], D, param[3], s)).fit(disp=-1)
        except Exception as e:
            print(e)
            continue
        aic = model.aic
        # saving best model, AIC and parameters
        if aic < best_aic:
            best_model = model
            best_aic = aic
            best_param = param
        myresults.append([param, model.aic])


    myresult_table = pd.DataFrame(myresults,columns = ['parameters', 'aic'])
    # sorting in ascending order, the lower AIC is - the better
    myresult_table = myresult_table.sort_values(by='aic', ascending=True).reset_index(drop=True)

    return myresult_table



# In[
from itertools import product
ps = range(0, 3)
d=0
qs = range(0, 3)
Ps = range(0, 3)
D=1
Qs = range(0, 3)
s = 24
parameters = product(ps, qs, Ps, Qs)
listparam = list(parameters)
len(listparam)
data = hourlyall['1/1/2018':'31/12/2018']


#a=optimizeSARIMA(data.nb, listparam, d, D, s)
model=SARIMA(data,3,0,0,2,1,0,24)