
import plotly.plotly as py
import plotly.graph_objs as go
import pandas as pd
import numpy as np

df = pd.read_excel('/Users/bounouamustapha/Desktop/work/all_data.xlsx')

df.index = df['DATE_ARRIVEE']
del df['DATE_ARRIVEE']



def mean_absolute_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    x = np.mean(np.abs(y_true - y_pred))
    return x


def symetrique_mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    x = np.mean(np.abs((y_true - y_pred) / (y_pred + y_true))) * 200
    return x