import numpy as np


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    x = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return x


def symetrique_mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    x = np.mean(np.abs((y_true - y_pred) / (y_pred + y_true))) * 200
    return x


def mean_absolute_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    x = np.mean(np.abs(y_true - y_pred))
    return x


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
import matplotlib.pyplot as plt


# In[] code


def showByHour(x, y):
    data_hour = hourly[x:y]
    data = [
        go.Scatter(
            x=data_hour.index,
            y=data_hour
        )
    ]
    py.plot(data)


def getInfo(x, y):
    data_hour = hourly[x:y]
    mean = hourly.resample('Y').mean()
    print("max: " + str(data_hour.max()))
    print("min: " + str(data_hour.min()))
    print("moyenne: " + str(mean))
    print("variance: " + str(np.var(hourly['nb'])))
    np.var


# In[
def showPerHourWeek(x, y):
    from pandas import DataFrame
    from datetime import timedelta
    data_hour_week = hourly[x:y]
    data_hour_week['day_of_week'] = data_hour_week.index.weekday_name
    start = data_hour_week.index[0]
    lines = []

    for i in range(0, 7):
        end = start + timedelta(days=1)
        f = data_hour_week[start:end]
        f['nb'].index = range(0, 25)
        dmd = DataFrame(index=f.index.hour, data=f['nb'])
        dmd = dmd.head(24)
        line = go.Scatter(
            x=np.arange(25),
            y=dmd['nb'],
            mode='lines',
            name=str(i)
        )
        lines.append(line)
        start = end
    py.plot(lines)
    return


# In[]
def getPerDayMonth(x, y):
    data_week = daily[x:y]
    data_week['day_of_week'] = data_week.index.weekday_name
    z = 0
    lines = []

    for i in range(0, len(data_week) - len(data_week) % 7):
        if i % 7 == 0:
            z += 1
            dd = data_week[i:i + 6]
            # dd = dd.groupby('day_of_week')['nb'].mean()
            line = go.Scatter(
                x=dd.index,
                y=dd,
                mode='lines',
                name=str(i)
            )
            lines.append(line)
    py.plot(lines)
    return


# In[]


def getMeanByWeek(x, y):
    data_week = hourly[x:y]
    wed = data_week.groupby([data_week.index.weekday_name, data_week.index.hour])['nb'].mean()
    lines = []
    for idate in wed.index.get_level_values(level=0).unique():
        line = go.Scatter(
            x=np.arange(0, 25),
            y=np.array(wed[idate]),
            mode='lines',
            name=idate
        )
        lines.append(line)
    py.plot(lines)
    return


# In[] code
def getMeanByWeek(x, y):
    data_week = hourly[x:y]
    wed = data_week.groupby([data_week.index.weekday_name, data_week.index.hour])['nb'].mean()
    lines = []
    for idate in wed.index.get_level_values(level=0).unique():
        line = go.Scatter(
            x=np.arange(0, 25),
            y=np.array(wed[idate]),
            mode='lines',
            name=idate
        )
        lines.append(line)
    py.plot(lines)
    return


# In[
def gethistoday(x, y):
    data_week = daily[x:y]

    wed = data_week.groupby([data_week.index.weekday_name])['nb'].mean()
    print(wed)
    line = go.Scatter(
        x=np.array(wed.index.get_level_values(level=0).unique()),
        y=np.array(wed),
        mode='lines',
    )
    py.plot([line], filename='basic-bar')
    return


# In[]:
def decompose(x, y):
    from plotly.plotly import plot_mpl
    from statsmodels.tsa.seasonal import seasonal_decompose
    h = hourly[x:y]
    result = seasonal_decompose(h, model='additive')
    fig = result.plot()
    plot_mpl(fig)



# In[] code
def arima(data, testdate, horizon, nbjourtest, seasonal, seasonality):
    from pyramid.arima import auto_arima
    from pyramid.arima import ARIMA
    from datetime import timedelta
    test_date_time = datetime.strptime(testdate, '%d/%m/%Y')
    end_test = test_date_time + timedelta(hours=horizon - 1)
    end_train = test_date_time - timedelta(1)
    start_train = test_date_time - timedelta(hours=nbjourtest)

    train = data[start_train:end_train]
    test = data[test_date_time:end_test]
    print('training set :' + str(start_train) + ' au ' + str(end_train))
    print('test set :' + str(test_date_time) + ' au ' + str(end_test))

    arima_model = auto_arima(train, seasonal=True, m=24, error_action='ignore', trace=1, stepwise=True)
    # arima_model = auto_arima(train,seasonal=seasonal, m=seasonality,error_action='ignore')

    prevision = arima_model.predict(horizon)
    precision = mean_absolute_error(test, prevision)
    print(arima_model.summary())

    print('-----------------------------------------------------------------------------')
    print('--------Mae : --------' + str(precision) + '--------------------------------')
    x = hourly[start_train:end_test]
    return prevision
    plt.plot(x.index, x)
    plt.plot(test.index, prevision)
    plt.legend(['observation', 'prevision'])
    plt.title('La prevision sur un horizon de  :' + str(horizon))
    plt.show()


# In[] test
def acf(y,lag):
    import statsmodels.tsa.api as smt
    smt.graphics.plot_acf(y, lags=lag)
    plt.show()


# In[] test

def pacf(y,lag):
    import statsmodels.tsa.api as smt
    smt.graphics.plot_pacf(y, lags=lag)
    plt.show()


# In[] test
def seasonal_diff(y,lag):
    import statsmodels.tsa.api as smt
    y_diff = y - y.shift(24)
    smt.graphics.plot_acf(y_diff, lags=lag)
    tsplot(y_diff[24:], lag)
    return y_diff[24:]







# In[] test






# In[] test
def adf(y):
    import statsmodels.api as sm
    import statsmodels.tsa.api as smt
    y = y - y.shift(1)
    y = y[24:]
    result=sm.tsa.stattools.adfuller(y)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
    return result

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
def sarima(data,p,d,q,P,D,Q,s):
    import statsmodels.api as sm
    model = sm.tsa.statespace.SARIMAX(data, order=(p, d, q),
                                           seasonal_order=
                                           (P, D, Q, s)).fit(disp=-1)
    print(model.summary())


# In[
h = hourly['1/1/2018':'31/1/2018']
sarima(h, 0, 0, 3, 0, 1, 2, 24)


