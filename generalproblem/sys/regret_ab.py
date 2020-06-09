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
costour_ab = np.zeros(3)
costrl_ab = np.zeros(3)
costmpc_ab = np.zeros(3)
costnos_ab = np.zeros(3)
costthb_ab = np.zeros(3)

for BAR_PAR in [0.5,1,3]:
    B = int(8820*BAR_PAR)
    costour = np.load('./data/costour_sys'+str(B)+'.npy')
    costrl = np.load('./data/costrl_sys'+str(B)+'.npy')
    costmpc = np.load('./data/costmpc_sys'+str(B)+'.npy')
    costnos = np.load('./data/costnos_sys'+str(B)+'.npy')
    costthb = np.load('./data/costthb_sys'+str(B)+'.npy')
    costofl = np.load('./data/costofl_sys'+str(B)+'.npy')
    
    costour = costour*costofl
    costrl = costrl*costofl
    costmpc = costmpc*costofl
    costnos = costnos*costofl
    costthb = costthb*costofl
    
    costour_ab[s] = np.percentile(costour[-1,:],50)
    costrl_ab[s] = np.percentile(costrl[-1,:],50)
    costmpc_ab[s] = np.percentile(costmpc[-1,:],50)
    costnos_ab[s] = np.percentile(costnos[-1,:],50)
    costthb_ab[s] = np.percentile(costthb[-1,:],50)
    s = s+1
    
np.save('./data/costour_ab.npy',costour_ab)
np.save('./data/costrl_ab.npy',costrl_ab)
np.save('./data/costmpc_ab.npy',costmpc_ab)
np.save('./data/costnos_ab.npy',costnos_ab)
np.save('./data/costthb_ab.npy',costthb_ab)
        
    
        



