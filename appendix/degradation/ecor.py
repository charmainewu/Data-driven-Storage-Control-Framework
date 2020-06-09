# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 11:00:26 2020

@author: jmwu
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from RL_brain import QLearningTable
from SCRFramework import BRFramework
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']

###############################read data###################################
df = pd.read_csv('YearData.csv')
D = df['demand'].values; 
P = df['Price Data'].values; 

uni_cost = 150000; deg = 0.01; L = 35;
ETA_C = 0.99; ETA_D = 1.01; MU = 0.001;
B = 8820; XIN = 0.7; FUL = 1; W = 2; step = 500;
LEVEL= np.array([20,2000,2000]) 
RL = QLearningTable(actions=list(range(3)))
BFR = BRFramework(XIN, FUL, LEVEL, B, W, ETA_C, ETA_D, MU)


print('###################################################')
B_size = BFR.battery_nodegradation(D,P,step,L,uni_cost)
print('###################################################')
B_size_d = BFR.battery_degradation(D,P,step,L,uni_cost,deg)

    
    
    
   