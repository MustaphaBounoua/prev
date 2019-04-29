# In[
import plotly.plotly as py
import plotly.graph_objs as go
import pandas as pd
import numpy as np

df = pd.read_excel('/Users/bounouamustapha/Desktop/work/all_data.xlsx')

df.index = df['DATE_ARRIVEE']
del df['DATE_ARRIVEE']


# In[
def mean_absolute_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    x = np.mean(np.abs(y_true - y_pred))
    return x

def symetrique_mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    x = np.mean(np.abs((y_true - y_pred) / (y_pred + y_true))) * 200
    return x


# In[
def seasonal_diff(y,nb,lag):
    import statsmodels.tsa.api as smt
    y_diff = y - y.shift(nb)
    tsplot(y_diff[nb:], lag)
    return y_diff[nb:]


# In[
def tsplot(y, lags=None, figsize=(12, 7), style='bmh'):
    import statsmodels.api as sm
    import statsmodels.tsa.api as smt
    import matplotlib.pyplot as plt
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
    model = sm.tsa.statespace.SARIMAX(data, order = (p, d, q),
                                           seasonal_order = (P, D, Q, s)).fit(disp=-1)
    print(model.summary())
    return model

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
data = df['7/1/2017':'30/6/2018']

# In[

data = df['1/1/2018':'31/1/2018']

# In[

data = df['1/4/2018':'30/4/2018']

# In[
#data.to_csv('/Users/bounouamustapha/Desktop/work/data_train.csv', encoding='utf-8')

seasonal_diff(data['nb'], 24, 6)

#tsplot(data['nb'],24*3)




# In[
def plotSARIMA(series, model, n_steps,s,d):
    """
            Plots model vs predicted values

            series - dataset with timeseries
            model - fitted SARIMA model
            n_steps - number of steps to predict in the future
    """
    import matplotlib.pyplot as plt
    # adding model values
    dat = series.copy()
    dat.columns = ['actual']
    dat['arima_model'] = model.fittedvalues
    # making a shift on s+d steps, because these values were unobserved by the model
    # due to the differentiating
    dat['arima_model'][:s+d] = np.NaN
    print(dat.head())

    # forecasting on n_steps forward
    forecast = model.predict(start = dat.shape[0], end = dat.shape[0]+n_steps)
    forecast = dat.arima_model.append(forecast)
    # calculate error, again having shifted on s+d steps from the beginning
    error = symetrique_mean_absolute_percentage_error(dat['actual'][s+d:], dat['arima_model'][s+d:])

    plt.figure(figsize=(15, 7))
    plt.title("Mean Absolute Percentage Error: {0:.2f}%".format(error))
    plt.plot(forecast, color='r', label="model")
    plt.axvspan(dat.index[-1], forecast.index[-1], alpha=0.5, color='lightgrey')
    plt.plot(dat.actual, label="actual")
    plt.legend()
    plt.grid(True)
    plt.show()

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

#sarima(data, 3, 0, 0, 2, 1, 0, 24)

optimizeSARIMA(data,parameters_list,0,1,24)
# In[
mod=[]
# In[
m = sarima(data,3,0,3,0,1,2,24)

# In[
plotSARIMA(df['7/1/2018':'7/3/2018'], m , 24*2 , 24, 0)


# In[ daily
daily=df.resample('D').sum()
# In[ daily
seasonal_diff(daily['nb'], 7, 20)




# In[] test
def testarima(data, testdate, horizon, nbjourtest, p,d,q,P,D,Q,s):
    from datetime import timedelta
    from pandas import datetime
    import matplotlib.pyplot as plt
    test_date_time = datetime.strptime(testdate, '%d/%m/%Y')
    end_test = test_date_time + timedelta(days=horizon - 1)
    end_train = test_date_time - timedelta(1)
    start_train = test_date_time - timedelta(days=nbjourtest)
    train = data[start_train:end_train]
    test = data[test_date_time:end_test]
    arima_model = sarima(train,p,d,q,P,D,Q,s)
    prevision = arima_model.predict(start=train.shape[0],end=train.shape[0]+horizon*24)
    precision = mean_absolute_error(test, prevision)
    print(arima_model.summary())
    print('-------- Mae : --------' + str(precision) + '--------------------------------------')
    plt.plot(test.index, test)
    plt.plot(test.index, prevision)
    plt.legend(['observation', 'prevision'])
    plt.title('La prevision sur un horizon de  :' + str(horizon)+'jour')
    plt.show()
    return arima_model


# In[

model = testarima(df, '10/2/2018', 3 , 365 , 2, 0, 3, 0, 1, 2, 24)





# In[
def test(testdate):
    from sklearn.metrics import mean_squared_error
    from math import sqrt
    p = 3
    d = 0
    q = 3
    P = 0
    D = 1
    Q = 2
    s = 24
    horizon = 3
    nbjourtest = 30
    from datetime import timedelta
    from pandas import datetime
    test_date_time = datetime.strptime(testdate, '%d/%m/%Y')
    end_test = test_date_time + timedelta(days=horizon)
    end_train = test_date_time
    start_train = test_date_time - timedelta(days=nbjourtest)
    train = df[start_train:end_train]
    train.drop(train.tail(1).index, inplace=True)
    test = df[test_date_time:end_test]
    test.drop(test.tail(1).index, inplace=True)

    arima_model = sarima(train, p, d, q, P, D, Q, s)
    prevision = arima_model.predict(start=train.shape[0], end=train.shape[0] -1 + horizon * 24)

    mae = mean_absolute_error(test.nb, prevision)

    return mae


# In[
from datetime import timedelta
from pandas import datetime

#startdata = range(365)
result = []
inter = []
k = 0
while k < 50:
    date = datetime.strptime('1/7/2018', '%d/%m/%Y')
    if k != 0:
        date = date + timedelta(days=k)
    precision = test(date.strftime('%d/%m/%Y'))
    print(str(date.strftime('%d/%m/%Y')))
    inter.append(precision)
    k = k + 13
    result.append(inter)





# In[ test arimmmmma
p=3
d=0
q=3
P=0
D=1
Q=2
s=24
testdate='22/7/2018'
horizon=3
nbjourtest=30
from datetime import timedelta
from pandas import datetime
import matplotlib.pyplot as plt
test_date_time = datetime.strptime(testdate, '%d/%m/%Y')
end_test = test_date_time + timedelta(days=horizon)
end_train = test_date_time
start_train = test_date_time - timedelta(days=nbjourtest)
train = df[start_train:end_train]
train.drop(train.tail(1).index,inplace=True)
test = df[test_date_time:end_test]
test.drop(test.tail(1).index,inplace=True)


# In[
arima_model = sarima(train,p,d,q,P,D,Q,s)

# In[


prevision = arima_model.predict(start=train.shape[0],end=train.shape[0]+3*24)
prevision.drop(prevision.tail(1).index,inplace=True)
precision = mean_absolute_error(test.nb, prevision)





# In[
end_test = test_date_time + timedelta(days=3)
test = df[test_date_time:end_test]
index = np.arange(3*24+1)
print(arima_model.summary())
print('-------- Mae : --------' + str(precision) + '--------------------------------------')
plt.plot(index, test)
plt.plot(index, prevision)
plt.legend(['observation', 'prevision'])
plt.title('La prevision sur un horizon de  :' + str(horizon)+'jour')
plt.show()
# In[

tsplot(arima_model.resid,24)


# In[
r = result[0]
tab =[]
for m in r:
    tab.append(m[0])

# In[
prevision.to_csv('/Users/bounouamustapha/Desktop/test2.csv')