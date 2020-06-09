# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 11:43:33 2020

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
df = pd.read_csv('YearData.csv')
D = df['demand'].values; 
P = df['Price Data'].values; 

for BAR_POR in [1]:
    B = 8820*BAR_POR; XIN = 0.7; FUL = 1; W = 2; 
    LEVEL= np.array([20,2000,2000]) 
    RL = QLearningTable(actions=list(range(3)))
    BF = BFramework(XIN, FUL, LEVEL, B, W)
    socour_o,socrl_o,socmpc_o,socnos_o,socthb_o,socofl_o = BF.general_performance(D,P)
    
    socour_ori = socour_o[0,:,:]
    socrl_ori = socrl_o[0,:,:]
    socmpc_ori = socmpc_o[0,:,:]
    socnos_ori = socnos_o[0,:,:]
    socthb_ori = socthb_o[0,:,:]
    socofl_ori = socofl_o[0,:,:]
    
    socour = np.zeros((731,2))
    socrl = np.zeros((731,2))
    socmpc = np.zeros((731,2))
    socnos = np.zeros((731,2))
    socofl = np.zeros((731,2))
    socthb = np.zeros((731,2))
    
    t= 24*5; T=24;
    for i in range(t,t+T):
        ####level
        socour[i,0] = socour_ori[i,0]
        socrl[i,0] = socrl_ori[i,0]
        socnos[i,0] = socnos_ori[i,0]
        socofl[i,0] = socofl_ori[i,0]
        socthb[i,0] = socthb_ori[i,0]
        socmpc[i,0] = socmpc_ori[i,0]
        ####decison
        socour[i,1] = max(min(1,socour_ori[i,3]-socour_ori[i,1]),-1)
        socrl[i,1] = max(min(1,socrl_ori[i,3]-socrl_ori[i,1]),-1)
        socnos[i,1] = max(min(1,socnos_ori[i,3]-socnos_ori[i,1]),-1)
        socofl[i,1] = max(min(1,socofl_ori[i,3]-socofl_ori[i,1]),-1)
        socthb[i,1] = max(min(1,socthb_ori[i,3]-socthb_ori[i,1]),-1)
        socmpc[i,1] = max(min(1,socmpc_ori[i,3]-socmpc_ori[i,1]),-1)
    
    x = np.linspace(1,T,T)
    fig, axs = plt.subplots(figsize=(10, 5))
    axs.step(x,P[t:t+T], linewidth = 3, marker = 'o', markersize = 1, c = 'lightslategrey', label = "Price")
    axs.legend(fontsize=15) 
    axs.set_xlabel('Time/h',fontsize=15)
    axs.set_ylabel('Price/$/MWh',fontsize=15)
    axs.tick_params(labelsize=13)
    plt.savefig("./figure/real_price.pdf")
    
    fig, axs = plt.subplots(2,4,figsize=(20, 10))
    ######## Charge/Discharged Energy    
    axs[0,0].set_title('Energy Storage Level, DETA')
    axs[0,1].set_title('Energy Storage Level, RL')
    axs[0,2].set_title('Energy Storage Level, MPC')
    axs[0,3].set_title('Energy Storage Level, THB')
    
    axs[0,0].set_xlabel('Time/h')
    axs[0,1].set_xlabel('Time/h')
    axs[0,2].set_xlabel('Time/h')
    axs[0,3].set_xlabel('Time/h')

    axs[0,0].set_ylabel('Energy/MW')
    axs[0,1].set_ylabel('Energy/MW')
    axs[0,2].set_ylabel('Energy/MW')
    axs[0,3].set_ylabel('Energy/MW')
    
    axs[0,0].bar(x,socofl[t:t+T,0], linewidth = 3, fc = 'lightgray', alpha = 0.6, label = "OFL")
    axs[0,1].bar(x,socofl[t:t+T,0], linewidth = 3, fc = 'lightgray', alpha = 0.6, label = "OFL")
    axs[0,2].bar(x,socofl[t:t+T,0], linewidth = 3, fc = 'lightgray', alpha = 0.6, label = "OFL")
    axs[0,3].bar(x,socofl[t:t+T,0], linewidth = 3, fc = 'lightgray', alpha = 0.6, label = "OFL")
    
    axs[0,0].step(x,socour[t:t+T,0], linewidth = 3, marker = 'o', markersize = 1, c = 'lightsalmon', label = "DETA")
    axs[0,1].step(x,socrl[t:t+T,0], linewidth = 3,  marker = 's',markersize = 1, c = 'lightslategrey', label = "RL")
    axs[0,2].step(x,socmpc[t:t+T,0], linewidth = 3,  marker = '*', markersize = 1, c = 'darkseagreen', label = "MPC")
    axs[0,3].step(x,socthb[t:t+T,0], linewidth = 3, marker = 'v', markersize = 1, c = 'cadetblue', label = "THB")
    
    ######Energy Storage Level
    axs[1,0].set_title('Charging/Discharging Decision, DETA')
    axs[1,1].set_title('Charging/Discharging Decision, RL')
    axs[1,2].set_title('Charging/Discharging Decision, MPC')
    axs[1,3].set_title('Charging/Discharging Decision, THB')
    
    axs[1,0].set_xlabel('Time/h')
    axs[1,1].set_xlabel('Time/h')
    axs[1,2].set_xlabel('Time/h')
    axs[1,3].set_xlabel('Time/h')
    
    axs[1,0].set_ylabel('Purchase Decision')
    axs[1,1].set_ylabel('Purchase Decision')
    axs[1,2].set_ylabel('Purchase Decision')
    axs[1,3].set_ylabel('Purchase Decision')
 
    
    axs[1,0].bar(x,socofl[t:t+T,1], linewidth = 3, fc = 'lightgray', alpha = 0.6, label = "OFL")
    axs[1,1].bar(x,socofl[t:t+T,1], linewidth = 3, fc = 'lightgray', alpha = 0.6, label = "OFL")
    axs[1,2].bar(x,socofl[t:t+T,1], linewidth = 3, fc = 'lightgray', alpha = 0.6, label = "OFL")
    axs[1,3].bar(x,socofl[t:t+T,1], linewidth = 3, fc = 'lightgray', alpha = 0.6, label = "OFL")
    
    axs[1,0].step(x,socour[t:t+T,1], linewidth = 3, marker = 'o', markersize = 1, c = 'lightsalmon', label = "DETA")
    axs[1,1].step(x,socrl[t:t+T,1], linewidth = 3,  marker = 's',markersize = 1, c = 'lightslategrey', label = "RL")
    axs[1,2].step(x,socmpc[t:t+T,1], linewidth = 3,  marker = '*', markersize = 1, c = 'darkseagreen', label = "MPC")
    axs[1,3].step(x,socthb[t:t+T,1], linewidth = 3, marker = 'v', markersize = 1, c = 'cadetblue', label = "THB")
    
    axs[1,0].set_yticks([-1,0,+1])
    axs[1,0].set_yticklabels((r'$-$',r'$0$',r'$+$'))
    axs[1,1].set_yticks([-1,0,+1])
    axs[1,1].set_yticklabels((r'$-$',r'$0$',r'$+$'))
    axs[1,2].set_yticks([-1,0,+1])
    axs[1,2].set_yticklabels((r'$-$',r'$0$',r'$+$'))
    axs[1,3].set_yticks([-1,0,+1])
    axs[1,3].set_yticklabels((r'$-$',r'$0$',r'$+$'))
    
    plt.savefig("./figure/real_perfor.pdf")