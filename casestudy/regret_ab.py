# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 18:47:09 2020

@author: jmwu
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from RL_brain import QLearningTable
from SCFramework import BFramework
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']

###############################read data###################################
s = 0 
costceta_ab = np.zeros(2)
costdeta_ab = np.zeros(2)
costpeta_ab = np.zeros(2)

for BAR_PAR in [0.5,1]:
    B = 8820*BAR_PAR
    costceta = np.load('./data/costceta'+str(B)+'.npy')
    costdeta = np.load('./data/costdeta'+str(B)+'.npy')
    costpeta = np.load('./data/costpeta'+str(B)+'.npy')
    costofl = np.load('./data/costdofl'+str(B)+'.npy')
    
    costceta = costceta*costofl
    costdeta = costdeta*costofl
    costpeta = costpeta*costofl
    
    costceta_ab[s] = np.percentile(costceta[-1,:],50)
    costdeta_ab[s] = np.percentile(costdeta[-1,:],50)
    costpeta_ab[s] = np.percentile(costpeta[-1,:],50)
   
    s = s+1
    
np.save('./data/costceta_ab.npy',costceta_ab)
np.save('./data/costdeta_ab.npy',costdeta_ab)
np.save('./data/costpeta_ab.npy',costpeta_ab)

        
    
        



