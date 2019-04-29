# In[
import plotly.plotly as py
import plotly.graph_objs as go
import pandas as pd
import numpy as np

df = pd.read_excel('/Users/bounouamustapha/Desktop/work/all_data.xlsx')

df.index = df['DATE_ARRIVEE']
del df['DATE_ARRIVEE']

daily=df.resample('D').sum()

def mean_absolute_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    x = np.mean(np.abs(y_true - y_pred))
    return x
def symetrique_mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    x = np.mean(np.abs((y_true - y_pred) / (y_pred + y_true))) * 200
    return x
# In[
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    x = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
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
                                           seasonal_order = (P, D, Q, s),enforce_stationarity=True).fit(disp=-1)
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
    prevision = arima_model.predict(start=train.shape[0],end=train.shape[0]+horizon)
    prevision.drop(prevision.tail(1).index, inplace=True)
    precision = mean_absolute_percentage_error(test.nb, prevision)

    #print(arima_model.summary())
    # print('-------- Mape : --------' + str(precision) + '--------------------------------------')
    #  plt.plot(test.index, test)
    # plt.plot(test.index, prevision)
    # plt.legend(['observation', 'prevision'])
    # plt.title('La prevision sur un horizon de  :' + str(horizon))
    #plt.show()
    return precision


# In[








# In[

m=testarima(daily,'22/7/2018', 2,365,2,0,2,0,1,1,7)

# In[
from datetime import timedelta
from pandas import datetime

testdata = [365]
startdata = range(365)
result = []

for i in testdata:
    inter = []
    k=0
    while k<180 :
        date = datetime.strptime('1/1/2018', '%d/%m/%Y')
        if k != 0:
            date = date + timedelta(days=k)
        pourcentage = testarima(daily, date.strftime('%d/%m/%Y'), 4, i, 2, 0, 2, 0, 1, 2, 7)
        inter.append(pourcentage)
        k=k+4
    result.append(inter)


# In[
mydata=daily['1/1/2018':'31/12/2018']
mydata.to_csv('/Users/bounouamustapha/Desktop/dataold.csv')

# In[
#tsplot(mydata.nb,30)
#seasonal_diff(mydata['nb'],7,20)
m=sarima(daily['7/1/2017':'6/30/2018'],2,0,2,0,1,1,7)
print(m.summary())
tsplot(m.resid[7:],30)

# In[
tsplot(daily['1/1/2018':'31/12/2018']['nb'],lags=21)