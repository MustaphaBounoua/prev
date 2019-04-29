

#import all necessary packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.tsa.api as smt


#importing data
def getHourlyData(filename='/Users/bounouamustapha/Desktop/work/all_data.xlsx',daily=False):
    df = pd.read_excel(filename)
    df.index = df['DATE_ARRIVEE']
    del df['DATE_ARRIVEE']
    if daily:
        return df.resample('D').sum()
    else:
        return df


# difference
def diff(y,nb,lag):
    y_diff = y - y.shift(nb)
    tsplot(y_diff[nb:], lag)
    return y_diff[nb:]

#SARIMAX


def sarima(data,p,d,q,P,D,Q,s):
    import statsmodels.api as sm
    model = sm.tsa.statespace.SARIMAX(data, order = (p, d, q),
                                           seasonal_order = (P, D, Q, s)).fit(disp=-1)
    print(model.summary())
    return model


#Optimize ARIMA
def optimizeARIMA(data,parameters_list, d, D, s):
    import statsmodels.api as sm
    results = []
    best_aic = float("inf")
    i=0
    for param in parameters_list:
        i+=1
        print("--"+ str(i + 1)+"/"+str(len(parameters_list))+"--")
        print("ARIMA "+ "("+str(param[0])+","+str(d)+","+str(param[1])+") ("+str(param[2])+","+str(D)+","+str(param[3])+")"+str(s))
        try:
            model = sm.tsa.statespace.SARIMAX(data, order=(param[0], d, param[1]),seasonal_order=(param[2], D, param[3], s)).fit(disp=-1)
            print("fitting")
            print("--------")
        except :
            print("Failed to estimate")
            print("------------------")
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
#accuracy functions


def mean_absolute_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    x = np.mean(np.abs(y_true - y_pred))
    return x


def symetrique_mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    x = np.mean(np.abs((y_true - y_pred) / (y_pred + y_true))) * 200
    return x


#Other functions
def tsplot(y, lags=None, figsize=(12, 7), style='bmh',adf = False):
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
        if adf:
            ts_ax.set_title('\n Augmented Dickey-Fuller: p={0:.5f}'.format(p_value))
        else:
            ts_ax.set_title('ACF et PACF')
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
        plt.tight_layout()
        plt.show()


def decompose(data):
    from plotly.plotly import plot_mpl
    from statsmodels.tsa.seasonal import seasonal_decompose
    h = data
    result = seasonal_decompose(h, model='additive')
    fig = result.plot()
    plot_mpl(fig)


def test_arima(data,testdate,nbjourpred,nbjourtest,p=3,d=0,q=3,P=0,D=1,Q=2,s=24):
    from sklearn.metrics import mean_squared_error
    from math import sqrt
    horizon = nbjourpred
    nbjourtest = nbjourtest
    from datetime import timedelta
    from pandas import datetime
    test_date_time = datetime.strptime(testdate, '%d/%m/%Y')
    end_test = test_date_time + timedelta(days=horizon)
    end_train = test_date_time
    start_train = test_date_time - timedelta(days=nbjourtest)
    train = data[start_train:end_train]
    train.drop(train.tail(1).index, inplace=True)
    test = data[test_date_time:end_test]
    test.drop(test.tail(1).index, inplace=True)
    arima_model = sarima(train, p, d, q, P, D, Q, s)
    prevision = arima_model.predict(start=train.shape[0], end=train.shape[0] -1 + horizon * 24)
    mae = mean_absolute_error(test.nb, prevision)
    rmse =  sqrt(mean_squared_error(test.nb,prevision))
    result = dict()
    result["Start Train"] = test_date_time
    result["End Train"] = end_train
    result['Start test'] = start_train
    result['End test'] = end_test
    result['predictions'] = prevision
    result["Horizon"] = horizon
    result["MAE"]= str(mae)
    result["rmse"] = str(rmse)
    return result


def cross_validation(data,start_date,horizon,nbjour_max,step,p=3,d=0,q=3,P=0,D=1,Q=2,s=24,train_size=range(365)):
    from datetime import timedelta
    from pandas import datetime

    result = []
    inter = []
    k = 0
    for m in train_size:
        while k < nbjour_max:
            date = datetime.strptime('1/7/2018', '%d/%m/%Y')
            if k != 0:
                date = date + timedelta(days=k)
            precision = test_arima(data.nb, date.strftime('%d/%m/%Y'), horizon,m ,3 , 0 , 3 , 0 , 1 , 2, 24)
            print(str(date.strftime('%d/%m/%Y')))
            inter.append(precision)
            k = k + step
            result.append(inter)