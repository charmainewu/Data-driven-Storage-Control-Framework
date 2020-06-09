# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 17:39:09 2020

@author: jmwu
"""

from scipy.stats import norm
import numpy as np
import scipy.integrate as integrate
import math
import itertools
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from scipy.stats import uniform
from scipy.stats import lognorm
import pandas as pd
from scipy import linalg
import matplotlib as mpl
from sklearn import mixture
import warnings
warnings.filterwarnings("ignore")
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
def estimate_bic(X):
    X = X.reshape(-1,1)
    bic = []; lowest_bic = np.infty;
    n_components_range = range(1, 10)
    cv_types = ['spherical']
    for cv_type in cv_types:
        for n_components in n_components_range:
            gmm = mixture.GaussianMixture(n_components=n_components,
                                          covariance_type=cv_type)
            gmm.fit(X)
            bic.append(gmm.bic(X))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm    
    clf = best_gmm
    
    bic = np.array(bic)
    color_iter = itertools.cycle(['lightslategrey'])        
    bars = []
    # Plot the BIC scores
    plt.figure(figsize=(8,2))
    spl = plt.subplot()
    for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
        xpos = np.array(n_components_range) + .2 * (i - 2)
        bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                      (i + 1) * len(n_components_range)],
                            width=.2, color=color))
    plt.xticks(n_components_range)
    plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
    plt.title('BIC score per model',fontsize=15)
    xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +\
        .2 * np.floor(bic.argmin() / len(n_components_range))
    plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=15)
    spl.set_xlabel('Number of components',fontsize=15)
    #spl.legend([b[0] for b in bars], cv_types,fontsize=15)
    spl.tick_params(labelsize=15)
    plt.savefig("bic.pdf")
    
    return clf


def estimate_gmm(X,n):
    X = X.reshape(-1,1)
    bic = []; lowest_bic = np.infty;
    n_components_range = range(n, n+1)
    cv_types = ['spherical']
    for cv_type in cv_types:
        for n_components in n_components_range:
            gmm = mixture.GaussianMixture(n_components=n_components,
                                          covariance_type=cv_type)
            gmm.fit(X)
            bic.append(gmm.bic(X))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm    
    clf = best_gmm
    
    return clf

df = pd.read_csv('RealData.csv')
d = df['Load Data']
p = df['Price Data']

estimate_bic(p.values)

fig, ax = plt.subplots(1, 1)
clf = estimate_gmm(p.values,1)
k,por,mean,std = len(clf.weights_),clf.weights_,clf.means_,clf.covariances_

x = np.linspace(0,99,100)
y = np.zeros(len(x))
for t in range(len(x)):
    for i in range(k):
        y[t] = y[t] + por[i]*norm.pdf(t,mean[i],np.sqrt(std[i]))
ax.plot(x, y, lw=3, dashes = [2,3] , color = 'lightsalmon',label = "GMM k=1")

clf = estimate_gmm(p.values,3)
k,por,mean,std = len(clf.weights_),clf.weights_,clf.means_,clf.covariances_

x = np.linspace(0,99,100)
y = np.zeros(len(x))
for t in range(len(x)):
    for i in range(k):
        y[t] = y[t] + por[i]*norm.pdf(t,mean[i],np.sqrt(std[i]))
ax.plot(x, y, lw=3, dashes = [1,0.5],color='darkseagreen', label = "GMM k=3")

clf = estimate_gmm(p.values,10)
k,por,mean,std = len(clf.weights_),clf.weights_,clf.means_,clf.covariances_

x = np.linspace(0,99,100)
y = np.zeros(len(x))
for t in range(len(x)):
    for i in range(k):
        y[t] = y[t] + por[i]*norm.pdf(t,mean[i],np.sqrt(std[i]))
ax.plot(x, y, lw=3, color='wheat', label = "GMM k=10")

###############################################################################
ax.hist(p, 30, density=True, histtype='stepfilled', alpha=0.2, color='lightslategrey', label = "Histogram of Price Data")
ax.legend(loc='best')
ax.tick_params(labelsize=15)
plt.xlabel('Price ($/MWh)',fontsize=15)
plt.ylabel("Density",fontsize=15)
plt.savefig("price.pdf")

