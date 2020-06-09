# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 18:59:48 2020

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

for BAR_POR in [0.5,1]:
    B = 8820*BAR_POR; XIN = 0.7; FUL = 1; W = 2;
    LEVEL= np.array([20,2000,2000]) 
    RL = QLearningTable(actions=list(range(3)))
    BF = BFramework(XIN, FUL, LEVEL, B, W)
    #ceta,peta,deta,ofl = BF.casestudy_value(df)
    
    ceta = np.load('./data/costceta'+str(B)+'.npy')
    peta = np.load('./data/costpeta'+str(B)+'.npy')
    deta = np.load('./data/costdeta'+str(B)+'.npy')
    dofl = np.load('./data/costdofl'+str(B)+'.npy')

    inteval = 5; ITER = 1; N = 168;
    
    x = list(range(inteval,N+1,inteval*1))
    name = ['Jan.','Feb.','Mar.','Apr.','May','Jun.','Jul.','Aug.','Sep.','Oct','Nov.','Dec.']
    color = ['lightgray','lightgray','lightgray','lightgray','darkseagreen','lightgray','lightsalmon','lightgray','lightgray','lightslategrey', 'lightgray','lightgray']
    ###############################################################################
    fig, axs = plt.subplots()

    for m in range(12):
        if m==4 or m==6 or m==9:
            axs.plot(x, deta[::1,m]-1, linewidth = 2.5,label = name[m],c=color[m])
        if m==1:
            axs.plot(x, deta[::1,m]-1, linewidth = 2.5,label = "Other Months", c=color[m])
        if m!=1 and m!=4 and m!=6 and m!=9:
            axs.plot(x, deta[::1,m]-1, linewidth = 2.5, c=color[m], alpha =0.5)

    plt.xlabel('Time/h',fontsize=15)
    plt.ylabel(r"$\gamma$",fontsize=15)
    plt.legend(loc=1, ncol= 2,fontsize=11)
    plt.savefig("./figure/deta"+str(B)+".pdf", dpi=1600)

    
    for month in range(12):
        x = list(range(inteval,N+1,inteval))
        fig, axs = plt.subplots()
        axs.plot(x, ceta[:,month]-1, linewidth = 2, markersize = 5,c='lightsalmon', marker='o',label = "DETA"+"$^{P}$")
        axs.plot(x, deta[:,month]-1, linewidth = 2,markersize = 5,c='lightslategrey', marker='s', label = "DETA")
        axs.plot(x, peta[:,month]-1, linewidth = 2,markersize = 5,c='darkseagreen', marker='v', label = "DETA"+"$^{I}$")
     
        axs.legend(fontsize=11)
        axs.set_xlabel('Time/h',fontsize=15)
        axs.set_ylabel(r"$\gamma$",fontsize=15)
        axs.tick_params(labelsize=15)
        plt.savefig("./figure/comp"+str(B)+str(month)+".pdf")
    
   