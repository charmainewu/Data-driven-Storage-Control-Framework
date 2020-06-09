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
    price_data['Date'][i] = data['Date'][i]
    price_data['Price'][i] = max(0,data['Price'][i])
price_data.index = price_data.Date
price_data.drop('Date', axis=1, inplace=True)
for i in range(0,len(data)):
    renewable_data['Date'][i] = data['Date'][i]
    renewable_data['Wind'][i] = data['Wind'][i]
renewable_data.index = renewable_data.Date
renewable_data.drop('Date', axis=1, inplace=True)
for i in range(0,len(data)):
    load_data['Date'][i] = data['Date'][i]
    load_data['Load'][i] = data['Load'][i]
load_data.index = load_data.Date
load_data.drop('Date', axis=1, inplace=True)
"""

df = pd.read_csv('RealData.csv')
df['Time'] = pd.to_datetime(df['TIME'], format='%m/%d/%Y %I:%M:%S %p')
df.index = df['Time']

D = df['DEMAND']; d = D[:168];
P = df['PRICE']; p = P[:168];


plt.figure(figsize=(30,17))
plt.subplot(2,1,1)
plt.plot(p, linewidth = 6, label='Price Series',c = 'lightslategrey')
plt.ylabel('Price $/MW',fontsize=35)
#plt.grid(True, linestyle=':')
#plt.grid(True, linestyle=':')
plt.legend(fontsize=40)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.subplot(2,1,2)
plt.plot(d, linewidth = 6,label='Demand Series',c = 'darkseagreen')
plt.ylabel('Demand/MW',fontsize=35)
plt.xlabel('Time',fontsize=35)
#plt.grid(True, linestyle=':')
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.legend(fontsize=40)

plt.savefig("prodata.pdf")