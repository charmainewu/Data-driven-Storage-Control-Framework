# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 13:28:57 2020

@author: jmwu
"""


from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import time
from datetime import datetime, date
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from seq2seq import SimpleSeq2Seq, Seq2Seq, AttentionSeq2Seq
from keras.models import load_model
from keras.callbacks import EarlyStopping 
import keras

##################################read data####################################
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
for i in range(0,len(data)):
    hour_data['Date'][i] = data['Date'][i]
    if data['Date'][i].hour<6:
        hour_data['Hour'][i] = 1
    elif data['Date'][i].hour>=6 and data['Date'][i].hour<12:
        hour_data['Hour'][i] = 2
    elif data['Date'][i].hour>=12 and data['Date'][i].hour<18:
        hour_data['Hour'][i] = 3
    else:
        hour_data['Hour'][i] = 4
hour_data.index = hour_data.Date
hour_data.drop('Date', axis=1, inplace=True)
for i in range(0,len(data)):
    month_data['Date'][i] = data['Date'][i]
    month_data['Month'][i] = data['Date'][i].month
month_data.index = month_data.Date
month_data.drop('Date', axis=1, inplace=True)

for i in range(0,len(data)):
    week_data['Date'][i] = data['Date'][i]
    week_data['Week'][i] = data['Date'][i].weekday()
week_data.index = week_data.Date
week_data.drop('Date', axis=1, inplace=True)

renewable_data = renewable_data["2017-01-01 00:00:00":"2018-12-31 23:00:00"]
price_data = price_data["2017-01-01 00:00:00":"2018-12-31 23:00:00"]
load_data = load_data["2017-01-01 00:00:00":"2018-12-31 23:00:00"]
hour_data = hour_data["2017-01-01 00:00:00":"2018-12-31 23:00:00"]
month_data = month_data["2017-01-01 00:00:00":"2018-12-31 23:00:00"]
week_data = week_data["2017-01-01 00:00:00":"2018-12-31 23:00:00"]

scaler_r = MinMaxScaler(feature_range=(0, 1))
r = scaler_r.fit_transform(renewable_data)

scaler_p = MinMaxScaler(feature_range=(0, 1))
p = scaler_p.fit_transform(price_data)

enc_h = OneHotEncoder(handle_unknown='ignore')
h = enc_h.fit_transform(hour_data).toarray()

enc_m = OneHotEncoder(handle_unknown='ignore')
m = enc_m.fit_transform(month_data).toarray()

enc_w = OneHotEncoder(handle_unknown='ignore')
w = enc_w.fit_transform(week_data).toarray()

##################################renewable pre####################################

BACK_DAY = 24;
dataset_r  = np.hstack((r,w,h))
train_size = int(len(dataset_r) * 0.50)
test_size = len(dataset_r) - train_size
train, test = dataset_r[0:train_size,:], dataset_r[train_size:len(dataset_r),:]
mae = [];rmse = []


for n_outputs in range(5):
    if n_outputs==0:
        n_outputs = 1
    else:    
        n_outputs = n_outputs*10
    
    x_train, y_train = [], []
    x_test, y_test = [], []
    
    for i in range(BACK_DAY,train_size-n_outputs):
        x_train.append(train[i-BACK_DAY:i,:])
        y_train.append(train[i:i+n_outputs,0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], 1))
    
    for i in range(BACK_DAY,test_size-n_outputs):
        x_test.append(test[i-BACK_DAY:i,:])
        y_test.append(test[i:i+n_outputs,0])
    x_test, y_test = np.array(x_test), np.array(y_test)
    y_test = y_test.reshape((y_test.shape[0], y_test.shape[1], 1))
    
    es = keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=0, 
    verbose=0, 
    mode='auto'
    )
    
    model = Seq2Seq(output_dim=1, hidden_dim=16, output_length=n_outputs, input_shape=(x_train.shape[1], x_train.shape[2]), peek=False, depth=2,dropout=0.2)
    model.compile(loss='mse', optimizer='adam')
    model.fit(x_train, y_train, epochs=100, batch_size=100, validation_data=(x_test, y_test), verbose=2, callbacks=[es])
    model.save_weights("./winddata/s2s"+str(n_outputs)+".h5")
    
    test_predict = model.predict(x_test)
    test_predict = test_predict.reshape((test_predict.shape[0], test_predict.shape[1]))
    test_predict = scaler_r.inverse_transform(test_predict)
    y_test = y_test.reshape((y_test.shape[0], y_test.shape[1]))
    y_test = scaler_r.inverse_transform(y_test)
    
    mae.append(mean_absolute_error(y_test[:600,:], test_predict[:600,:]))
    rmse.append(np.sqrt(mean_squared_error(y_test[:600,:], test_predict[:600,:])))
    print('Test Mean Absolute Error:', mean_absolute_error(y_test[:600,:], test_predict[:600,:]))
    print('Test Root Mean Squared Error:',np.sqrt(mean_squared_error(y_test[:600,:], test_predict[:600,:])))
    del model
    
mae,rmse = np.array(mae), np.array(rmse) 
np.save("./winddata/s2smae.npy",mae)
np.save("./winddata/s2srmse.npy",rmse)




