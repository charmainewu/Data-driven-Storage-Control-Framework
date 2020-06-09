# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 15:02:08 2020

@author: jmwu
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import matplotlib.gridspec as gridspec
warnings.filterwarnings("ignore")
from RL_brain import QLearningTable
from SCFramework import BFramework
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']

###############################read data###################################
df = pd.read_csv('YearData.csv')
D = df['demand'].values
P = df['Price Data'].values

for BAR_POR,W in [[0.5,2]]:
    B = int(8820*BAR_POR); XIN = 0.7; FUL = 1; 
    LEVEL= np.array([20,2000,2000]) 
    RL = QLearningTable(actions=list(range(3)))
    BF = BFramework(XIN, FUL, LEVEL, B, W)
    #costour,costrl,costmpc,costnos,costthb,costofl = BF.general_ratio_sys(D,P)
    
    costour = np.load('./data/costour_sys'+str(B)+'.npy')
    costrl = np.load('./data/costrl_sys'+str(B)+'.npy')
    costmpc = np.load('./data/costmpc_sys'+str(B)+'.npy')
    costnos = np.load('./data/costnos_sys'+str(B)+'.npy')
    costthb = np.load('./data/costthb_sys'+str(B)+'.npy')
    costofl = np.load('./data/costofl_sys'+str(B)+'.npy')
    
    x = list(range(5,730+1,5))
    gs = gridspec.GridSpec(2, 1, height_ratios=[0.2, 0.8])
    fig = plt.figure()
    axs1 = fig.add_subplot(gs[0])
    axs2 = fig.add_subplot(gs[1])
    
    axs1.set_ylim(1, 1.5)  # outliers only
    axs2.set_ylim(0.0, 0.5)  # most of the data
    
    min_ser = [np.percentile(i, 10) for i in costour-1]
    max_ser = [np.percentile(i, 90) for i in costour-1]
    mean_ser = [np.mean(i) for i in costour-1]
    axs1.plot(x, mean_ser, linewidth = 2.5,  c = 'lightsalmon', label = "DETA")
    axs2.plot(x, mean_ser, linewidth = 2.5,  c = 'lightsalmon', label = "DETA")
    axs2.fill_between(x, min_ser, max_ser, alpha=0.2, color= 'lightsalmon', label = "Percentile 5%-95% for DETA" ) 
    
    min_ser = [np.percentile(i, 10) for i in costrl-1]
    max_ser = [np.percentile(i, 90) for i in costrl-1]
    mean_ser = [np.mean(i) for i in costrl-1]
    axs1.plot(x, mean_ser, linewidth = 2.5,  c = 'lightslategrey', label = "RL")
    axs2.plot(x, mean_ser, linewidth = 2.5,  c = 'lightslategrey', label = "RL")
    axs2.fill_between(x, min_ser, max_ser, alpha=0.2, color= 'lightslategrey', label = "Percentile 5%-95% for RL" )
    
    min_ser = [np.percentile(i, 10) for i in costmpc-1]
    max_ser = [np.percentile(i, 90) for i in costmpc-1]
    mean_ser = [np.mean(i) for i in costmpc-1]
    axs1.plot(x, mean_ser, linewidth = 2.5,  c = 'darkseagreen', label = "MPC")
    axs2.plot(x, mean_ser, linewidth = 2.5,  c = 'darkseagreen', label = "MPC")
    axs2.fill_between(x, min_ser, max_ser, alpha=0.2, color = 'darkseagreen', label = "Percentile 5%-95% for MPC" )
    
    min_ser = [np.percentile(i, 10) for i in costthb-1]
    max_ser = [np.percentile(i, 90) for i in costthb-1]
    mean_ser = [np.mean(i) for i in costthb-1]
    axs1.plot(x, mean_ser, linewidth = 2.5,   c = 'wheat',label = "THB")
    axs2.plot(x, mean_ser, linewidth = 2.5,   c = 'wheat',label = "THB")
    axs2.fill_between(x, min_ser, max_ser, alpha=0.2, color= 'wheat', label = "Percentile 5%-95% for THB" )

    min_ser = [np.percentile(i, 10) for i in costnos-1]
    max_ser = [np.percentile(i, 90) for i in costnos-1]
    mean_ser = [np.mean(i) for i in costnos-1]
    axs1.plot(x, mean_ser, linewidth = 2.5,  c = 'lightgray', label = "BB")
    axs2.plot(x, mean_ser, linewidth = 2.5,  c = 'lightgray', label = "BB")
    axs2.fill_between(x, min_ser, max_ser, alpha=0.2, color = 'lightgray', label = "Percentile 5%-95% for NO STORAGE" )
    
    axs1.spines['bottom'].set_visible(False)
    axs2.spines['top'].set_visible(False)
    axs1.xaxis.tick_top()
    axs1.tick_params(labeltop=False)  # don't put tick labels at the top
    axs2.xaxis.tick_bottom()
    
    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=axs1.transAxes, color='k', clip_on=False)
    axs1.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    axs1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
    
    kwargs.update(transform=axs2.transAxes)  # switch to the bottom axes
    axs2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    axs2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
    
    axs1.legend(loc=0, ncol= 3, fontsize=15) 
    axs2.set_xlabel('Time/h',fontsize=15)
    axs2.set_ylabel(r"$\gamma$",fontsize=15)
    axs1.tick_params(labelsize=13)
    axs2.tick_params(labelsize=13)
    plt.savefig("./figure/rs"+str(B)+".pdf")
    
for BAR_POR,W in [[1,2]]:
    B = int(8820*BAR_POR); XIN = 0.7; FUL = 1; 
    LEVEL= np.array([20,2000,2000]) 
    RL = QLearningTable(actions=list(range(3)))
    BF = BFramework(XIN, FUL, LEVEL, B, W)
    #costour,costrl,costmpc,costnos,costthb,costofl = BF.general_ratio_sys(D,P)
    
    costour = np.load('./data/costour_sys'+str(B)+'.npy')
    costrl = np.load('./data/costrl_sys'+str(B)+'.npy')
    costmpc = np.load('./data/costmpc_sys'+str(B)+'.npy')
    costnos = np.load('./data/costnos_sys'+str(B)+'.npy')
    costthb = np.load('./data/costthb_sys'+str(B)+'.npy')
    costofl = np.load('./data/costofl_sys'+str(B)+'.npy')
    
    x = list(range(5,730+1,5))
    gs = gridspec.GridSpec(2, 1, height_ratios=[0.2, 0.8])
    fig = plt.figure()
    axs1 = fig.add_subplot(gs[0])
    axs2 = fig.add_subplot(gs[1])
    
    axs1.set_ylim(1.0, 1.5)  # outliers only
    axs2.set_ylim(0, 0.5)  # most of the data
    
    min_ser = [np.percentile(i, 10) for i in costour-1]
    max_ser = [np.percentile(i, 90) for i in costour-1]
    mean_ser = [np.mean(i) for i in costour-1]
    axs1.plot(x, mean_ser, linewidth = 2.5,  c = 'lightsalmon', label = "DETA")
    axs2.plot(x, mean_ser, linewidth = 2.5,  c = 'lightsalmon', label = "DETA")
    axs2.fill_between(x, min_ser, max_ser, alpha=0.2, color= 'lightsalmon', label = "Percentile 5%-95% for DETA" ) 
    
    min_ser = [np.percentile(i, 10) for i in costrl-1]
    max_ser = [np.percentile(i, 90) for i in costrl-1]
    mean_ser = [np.mean(i) for i in costrl-1]
    axs1.plot(x, mean_ser, linewidth = 2.5,  c = 'lightslategrey', label = "RL")
    axs2.plot(x, mean_ser, linewidth = 2.5,  c = 'lightslategrey', label = "RL")
    axs2.fill_between(x, min_ser, max_ser, alpha=0.2, color= 'lightslategrey', label = "Percentile 5%-95% for RL" )
    
    min_ser = [np.percentile(i, 10) for i in costmpc-1]
    max_ser = [np.percentile(i, 90) for i in costmpc-1]
    mean_ser = [np.mean(i) for i in costmpc-1]
    axs1.plot(x, mean_ser, linewidth = 2.5,  c = 'darkseagreen', label = "MPC")
    axs2.plot(x, mean_ser, linewidth = 2.5,  c = 'darkseagreen', label = "MPC")
    axs2.fill_between(x, min_ser, max_ser, alpha=0.2, color = 'darkseagreen', label = "Percentile 5%-95% for MPC" )
    
    min_ser = [np.percentile(i, 10) for i in costthb-1]
    max_ser = [np.percentile(i, 90) for i in costthb-1]
    mean_ser = [np.mean(i) for i in costthb-1]
    axs1.plot(x, mean_ser, linewidth = 2.5,   c = 'wheat',label = "THB")
    axs2.plot(x, mean_ser, linewidth = 2.5,   c = 'wheat',label = "THB")
    axs2.fill_between(x, min_ser, max_ser, alpha=0.2, color= 'wheat', label = "Percentile 5%-95% for THB" )

    min_ser = [np.percentile(i, 10) for i in costnos-1]
    max_ser = [np.percentile(i, 90) for i in costnos-1]
    mean_ser = [np.mean(i) for i in costnos-1]
    axs1.plot(x, mean_ser, linewidth = 2.5,  c = 'lightgray', label = "BB")
    axs2.plot(x, mean_ser, linewidth = 2.5,  c = 'lightgray', label = "BB")
    axs2.fill_between(x, min_ser, max_ser, alpha=0.2, color = 'lightgray', label = "Percentile 5%-95% for NO STORAGE" )
    
    axs1.spines['bottom'].set_visible(False)
    axs2.spines['top'].set_visible(False)
    axs1.xaxis.tick_top()
    axs1.tick_params(labeltop=False)  # don't put tick labels at the top
    axs2.xaxis.tick_bottom()
    
    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=axs1.transAxes, color='k', clip_on=False)
    axs1.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    axs1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
    
    kwargs.update(transform=axs2.transAxes)  # switch to the bottom axes
    axs2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    axs2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
    
    axs1.legend(loc=0, ncol= 3, fontsize=15) 
    axs2.set_xlabel('Time/h',fontsize=15)
    axs2.set_ylabel(r"$\gamma$",fontsize=15)
    axs1.tick_params(labelsize=13)
    axs2.tick_params(labelsize=13)
    plt.savefig("./figure/rs"+str(B)+".pdf")


for BAR_POR,W in [[3,3]]:
    B = int(8820*BAR_POR); XIN = 0.7; FUL = 1; 
    LEVEL= np.array([20,2000,2000]) 
    RL = QLearningTable(actions=list(range(3)))
    BF = BFramework(XIN, FUL, LEVEL, B, W)
    #costour,costrl,costmpc,costnos,costthb,costofl = BF.general_ratio_sys(D,P)
    
    costour = np.load('./data/costour_sys'+str(B)+'.npy')
    costrl = np.load('./data/costrl_sys'+str(B)+'.npy')
    costmpc = np.load('./data/costmpc_sys'+str(B)+'.npy')
    costnos = np.load('./data/costnos_sys'+str(B)+'.npy')
    costthb = np.load('./data/costthb_sys'+str(B)+'.npy')
    costofl = np.load('./data/costofl_sys'+str(B)+'.npy')
    
    x = list(range(5,730+1,5))
    gs = gridspec.GridSpec(2, 1, height_ratios=[0.2, 0.8])
    fig = plt.figure()
    axs1 = fig.add_subplot(gs[0])
    axs2 = fig.add_subplot(gs[1])
    
    axs1.set_ylim(1.0, 1.5)  # outliers only
    axs2.set_ylim(0, 0.5)  # most of the data
    
    min_ser = [np.percentile(i, 10) for i in costour-1]
    max_ser = [np.percentile(i, 90) for i in costour-1]
    mean_ser = [np.mean(i) for i in costour-1]
    axs1.plot(x, mean_ser, linewidth = 2.5,  c = 'lightsalmon', label = "DETA")
    axs2.plot(x, mean_ser, linewidth = 2.5,  c = 'lightsalmon', label = "DETA")
    axs2.fill_between(x, min_ser, max_ser, alpha=0.2, color= 'lightsalmon', label = "Percentile 5%-95% for DETA" ) 
    
    min_ser = [np.percentile(i, 10) for i in costrl-1]
    max_ser = [np.percentile(i, 90) for i in costrl-1]
    mean_ser = [np.mean(i) for i in costrl-1]
    axs1.plot(x, mean_ser, linewidth = 2.5,  c = 'lightslategrey', label = "RL")
    axs2.plot(x, mean_ser, linewidth = 2.5,  c = 'lightslategrey', label = "RL")
    axs2.fill_between(x, min_ser, max_ser, alpha=0.2, color= 'lightslategrey', label = "Percentile 5%-95% for RL" )
    
    min_ser = [np.percentile(i, 10) for i in costmpc-1]
    max_ser = [np.percentile(i, 90) for i in costmpc-1]
    mean_ser = [np.mean(i) for i in costmpc-1]
    axs1.plot(x, mean_ser, linewidth = 2.5,  c = 'darkseagreen', label = "MPC")
    axs2.plot(x, mean_ser, linewidth = 2.5,  c = 'darkseagreen', label = "MPC")
    axs2.fill_between(x, min_ser, max_ser, alpha=0.2, color = 'darkseagreen', label = "Percentile 5%-95% for MPC" )
    
    min_ser = [np.percentile(i, 10) for i in costthb-1]
    max_ser = [np.percentile(i, 90) for i in costthb-1]
    mean_ser = [np.mean(i) for i in costthb-1]
    axs1.plot(x, mean_ser, linewidth = 2.5,   c = 'wheat',label = "THB")
    axs2.plot(x, mean_ser, linewidth = 2.5,   c = 'wheat',label = "THB")
    axs2.fill_between(x, min_ser, max_ser, alpha=0.2, color= 'wheat', label = "Percentile 5%-95% for THB" )

    min_ser = [np.percentile(i, 10) for i in costnos-1]
    max_ser = [np.percentile(i, 90) for i in costnos-1]
    mean_ser = [np.mean(i) for i in costnos-1]
    axs1.plot(x, mean_ser, linewidth = 2.5,  c = 'lightgray', label = "BB")
    axs2.plot(x, mean_ser, linewidth = 2.5,  c = 'lightgray', label = "BB")
    axs2.fill_between(x, min_ser, max_ser, alpha=0.2, color = 'lightgray', label = "Percentile 5%-95% for NO STORAGE" )
    
    axs1.spines['bottom'].set_visible(False)
    axs2.spines['top'].set_visible(False)
    axs1.xaxis.tick_top()
    axs1.tick_params(labeltop=False)  # don't put tick labels at the top
    axs2.xaxis.tick_bottom()
    
    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=axs1.transAxes, color='k', clip_on=False)
    axs1.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    axs1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
    
    kwargs.update(transform=axs2.transAxes)  # switch to the bottom axes
    axs2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    axs2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
    
    axs1.legend(loc=0, ncol= 3, fontsize=15) 
    axs2.set_xlabel('Time/h',fontsize=15)
    axs2.set_ylabel(r"$\gamma$",fontsize=15)
    axs1.tick_params(labelsize=13)
    axs2.tick_params(labelsize=13)
    plt.savefig("./figure/rs"+str(B)+".pdf")
    




