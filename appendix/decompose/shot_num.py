# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 10:54:23 2020

@author: jmwu
"""



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
###############################################################################
def isDecompose(at,B,x0):
    Acc_sum = 0
    Acc = np.zeros(len(at))
    for i in range(len(at)):
        Acc[i] = Acc_sum + at[i]
        Acc_sum = Acc_sum + at[i]
    AccB = Acc + B
    
    sadwAB = np.zeros(int(max(Acc)+1))
    
    for i in range(len(at)):
        if AccB[i] <= max(Acc):
            sadwAB[int(AccB[i])] = 1
        sadwAB[int(Acc[i])] = 1
    
    sadwA = np.zeros(int(max(Acc)+1))
    sadwB = np.zeros(int(max(Acc)+1))
    
    i = 0; j = int(Acc[0]); 
    while(j <= max(Acc) and i+1<=len(at)-1):
        while(i+1<=len(at)-1 and Acc[i+1]==Acc[i]):
            try:
                sadwA[j] = i+1
                i = i + 1
            except:
                break
        while(i+1<=len(at)-1 and Acc[i+1]>Acc[i]):
            k = Acc[i+1]-Acc[i]
            try:
                sadwA[j] = i + 1
            except: 
                break
            while(k > 0):
                j = j + 1
                try:
                    sadwA[j] = i + 1
                    k = k - 1
                except:
                    break
            i = i + 1
            
    i = 0; j = int(AccB[0]); 
    while(j <= max(Acc) and i+1<=len(at)-1):
        while(i+1<=len(at)-1 and AccB[i+1]==AccB[i]):
            i = i + 1
        while(i+1<=len(at)-1 and AccB[i+1]>AccB[i]):
            k = AccB[i+1]-AccB[i]
            while(k > 0):
                j = j + 1
                try:
                    sadwB[j] = i + 1
                    k = k - 1
                except:
                    break
            i = i + 1
    
    a_index =np.where(sadwAB==1)[0]
    
    a = [];ts = [];tnz = [];
    for i in range(len(a_index)-1):
        a.append(a_index[i+1]-a_index[i])
        ts.append(sadwB[a_index[i+1]])
        tnz.append(sadwA[a_index[i]])
    
    Trunc_sum = a[0]; t = 0; del_list = [];
    while(Trunc_sum <= x0):
        del_list.append(t)
        t = t + 1
        try:
            Trunc_sum = Trunc_sum + a[t]
        except:
            break
                
    for i in del_list:
        del a[i]
        del ts[i]
        del tnz[i]
    a[0] = Trunc_sum - x0
    return a, ts, tnz


def decdata(df,month):
    dfm = df[df['Month'] == month]
    p = dfm['Price Data']
    a = dfm['Load Data (APT)']
    n = len(p)
    return a,p,n

###############################################################################
df = pd.read_csv('./YearData.csv')
###############################################################################
L = df['demand'].astype(int)
###############################################################################
B = int(max(L)*0.5)
a, ts, tnz = isDecompose(L,B,0)
shot_len = (np.array(tnz)-np.array(ts))

size, scale = 1000, 10
commutes = pd.Series(shot_len)

commutes.plot.hist(bins=7, rwidth=0.9,
                   color='#607c8e')
plt.xlabel('Shot Length')
plt.ylabel('Counts')
plt.grid(axis='y', alpha=0.75)

plt.savefig("./figure/b05.pdf")
plt.close()
###############################################################################
B = int(max(L))
a, ts, tnz = isDecompose(L,B,0)
shot_len = (np.array(tnz)-np.array(ts))

size, scale = 1000, 10
commutes = pd.Series(shot_len)

commutes.plot.hist(bins=7,rwidth=0.9,
                   color='#607c8e')
plt.xlabel('Shot Length')
plt.ylabel('Counts')
plt.grid(axis='y', alpha=0.75)
plt.savefig("./figure/b1.pdf")
plt.close()
###############################################################################
B = int(max(L)*2)
a, ts, tnz = isDecompose(L,B,0)
shot_len = (np.array(tnz)-np.array(ts))

size, scale = 1000, 10
commutes = pd.Series(shot_len)

commutes.plot.hist(bins=7,rwidth=0.9,
                   color='#607c8e')
plt.xlabel('Shot Length')
plt.ylabel('Counts')
plt.grid(axis='y', alpha=0.75)
plt.savefig("./figure/b2.pdf")
plt.close()
###############################################################################
B = int(max(L)*5)
a, ts, tnz = isDecompose(L,B,0)
shot_len = (np.array(tnz)-np.array(ts))

size, scale = 1000, 10
commutes = pd.Series(shot_len)

commutes.plot.hist(bins=7, rwidth=0.9,
                   color='#607c8e')
plt.xlabel('Shot Length')
plt.ylabel('Counts')
plt.grid(axis='y', alpha=0.75)
plt.savefig("./figure/b5.pdf")
###############################################################################
B = int(max(L)*10)
a, ts, tnz = isDecompose(L,B,0)
shot_len = (np.array(tnz)-np.array(ts))

size, scale = 1000, 10
commutes = pd.Series(shot_len)

commutes.plot.hist(bins=7,rwidth=0.9,
                   color='#607c8e')
plt.xlabel('Shot Length')
plt.ylabel('Counts')
plt.grid(axis='y', alpha=0.75)
plt.savefig("./figure/b10.pdf")
plt.close()
###############################################################################
B = int(max(L)*20)
a, ts, tnz = isDecompose(L,B,0)
shot_len = (np.array(tnz)-np.array(ts))

size, scale = 1000, 10
commutes = pd.Series(shot_len)

commutes.plot.hist(bins=7, rwidth=0.9,
                   color='#607c8e')
plt.xlabel('Shot Length')
plt.ylabel('Counts')
plt.grid(axis='y', alpha=0.75)
plt.savefig("./figure/b20.pdf")
plt.close()
###############################################################################
B = int(max(L)*50)
a, ts, tnz = isDecompose(L,B,0)
shot_len = (np.array(tnz)-np.array(ts))

size, scale = 1000, 10
commutes = pd.Series(shot_len)

commutes.plot.hist(bins=7, rwidth=0.9,
                   color='#607c8e')
plt.xlabel('Shot Length')
plt.ylabel('Counts')
plt.grid(axis='y', alpha=0.75)
plt.savefig("./figure/b50.pdf")
plt.close()







