# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 16:10:27 2020

@author: jmwu
"""

import warnings
warnings.filterwarnings("ignore")
from scipy.stats import norm
from sklearn import mixture
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime, date
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from keras.models import load_model
from seq2seq import SimpleSeq2Seq, Seq2Seq, AttentionSeq2Seq

from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
import matplotlib.pyplot as plt

#sns.set_style('white')
#sns.despine()

fs= 13
fs1 = 17
fs2= 15
###############################################################################
rs2smae = np.load("./winddata/s2smae.npy")/82000
rs2srmse = np.load("./winddata/s2srmse.npy")/82000
ds2smae = np.load("./loaddata/s2smae.npy")/82000
ds2srmse = np.load("./loaddata/s2srmse.npy")/82000
###############################################################################
x = [1,10,20,30,40]
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 5), sharey=True)
axs[0].plot(x, rs2smae, linewidth=2,marker = "*", markersize = 15, c = 'lightsalmon',label='Renewables Prediction')
axs[1].plot(x, ds2smae, linewidth=2, marker = "o",markersize = 10,c = 'darkseagreen',label='Demand Prediction')
#axs[1,0].plot(x, rs2srmse, linewidth=2, marker = "^",markersize = 10, c = 'darkseagreen',label='Renewables Prediction')
#axs[1,1].plot(x, ds2srmse, linewidth=2, marker = "s", markersize = 10, c = 'darkseagreen',label='Demand Prediction')
###############################################################################
axs[0].legend(fontsize=fs)
axs[1].legend(fontsize=fs)
#axs[1,0].legend(fontsize=fs)
#axs[1,1].legend(fontsize=fs)

axs[0].tick_params(labelsize=fs2)
axs[1].tick_params(labelsize=fs2)
#axs[1,0].tick_params(labelsize=fs2)
#axs[1,1].tick_params(labelsize=fs2)

axs[0].set_title('MAE, Renewables', fontsize=fs1)
axs[1].set_title('MAE, Demand', fontsize=fs1)
#axs[1,0].set_title('RMSE, Renewables', fontsize=fs1)
#axs[1,1].set_title('RMSE, Demand', fontsize=fs1)

axs[0].set_xlabel("# prediction steps",fontsize=fs1)
axs[0].set_ylabel("MAE",fontsize=fs1)
axs[1].set_xlabel("# prediction steps",fontsize=fs1)
axs[1].set_ylabel("MAE",fontsize=fs1)
plt.savefig("./figure/prederror.pdf",dpi=1600)