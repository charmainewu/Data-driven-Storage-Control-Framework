# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 17:24:50 2020

@author: jmwu
"""

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20,10
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
import matplotlib.pyplot as plt
"""
df = pd.read_csv('simulation.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%Y/%m/%d %H:%M')
df.index = df['Date']
data = df.sort_index(ascending=True, axis=0)

price_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Price'])
renewable_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Wind'])
load_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Load'])
hour_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Hour'])
week_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Week'])
month_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Month'])

for i in range(0,len(data)):
    renewable_data['Date'][i] = data['Date'][i]
    renewable_data['Wind'][i] = data['Wind'][i]
renewable_data.index = renewable_data.Date
renewable_data.drop('Date', axis=1, inplace=True)

renewable_data = renewable_data["2018-01-01 00:00:00":"2018-01-31 23:00:00"]
"""
plt.figure(figsize=(30,7))
plt.plot(renewable_data,linewidth = 6, label='Renewables Series',c = '#607c8e')
plt.xlabel('Time/h',fontsize=35)
plt.ylabel('Renewables/MW',fontsize=35)

plt.legend(fontsize=40)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.savefig("rewdata.pdf")