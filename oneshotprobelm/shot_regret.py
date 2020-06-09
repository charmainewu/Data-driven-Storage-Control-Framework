# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 20:38:13 2019

@author: lenovo
"""
#journal 1. shot with bound


from scipy.stats import norm
import numpy as np
import scipy.integrate as integrate
import math
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
mpl.rcParams["axes.prop_cycle"].by_key()["color"]=['lightslategrey', 'lightsalmon', 'darkseagreen', 'wheat']
#mpl.rcParams['axes.prop_cycle'] = ['lightslategrey', 'lightsalmon', 'darkseagreen', 'wheat']
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


def Athb(theta,B,t,at,pt,xt0,tnz):
    mu_c = 100; mu_d =100
    if t == tnz:
        dt =  min(at,mu_d,xt0)
        vat = at - dt
        vbt = 0
        xt = xt0+vbt-dt
        return xt,dt,vat,vbt
    if pt<=theta:
        dt = 0
        vat = at
        vbt = min(max(B-xt0,0),mu_c)
    else:
        dt =  min(at,mu_d,xt0)
        vat = at - dt
        vbt = 0
    xt = xt0+vbt-dt
    return xt,dt,vat,vbt
            
def Athb_ld(theta,ts,tnz,abar,pt,t,xt0):
    if t == tnz:
        abart = abar
    else:
        abart = 0
    return Athb(theta,abar,t,abart,pt,xt0,tnz)

def Athb_hat(ao,po,B):
    a = ao.copy() ; p = po.copy()
    x = np.zeros(int(len(a)))
    d = np.zeros(int(len(a)))
    va = np.zeros(int(len(a)))
    vb = np.zeros(int(len(a)))
    x0 = 0;
    abar, ts, tnz = isDecompose(a,B,x0)
    xi = np.zeros((len(abar),len(a)))
    cost_shot = np.zeros((len(abar),len(a)))
    theta = 0
    for t in range(len(a)):
        try:
            theta = math.sqrt(max(p[1:t])*min(p[1:t]))
        except:
            theta = theta
        for i in range(len(abar)):
            if t>=ts[i] and t<=tnz[i]:
                if t==ts[i]:
                    xt,dt,vat,vbt = Athb_ld(theta,ts[i],tnz[i],abar[i],p[t],
                                            t,0)
                    x[t] = x[t] + xt
                    d[t] = d[t] + dt
                    va[t] = va[t] + vat
                    vb[t] = vb[t] + vbt
                    a[t] = a[t] - dt - vat 
                    xi[i,t] = xt
                    if vat+vbt>0:
                        cost_shot[i,t] =  p[t]*(vat+vbt)
                else:
                    xt,dt,vat,vbt = Athb_ld(theta,ts[i],tnz[i],abar[i],p[t],
                                            t,xi[i,t-1])
                    x[t] = x[t] + xt
                    d[t] = d[t] + dt
                    va[t] = va[t] + vat
                    vb[t] = vb[t] + vbt
                    a[t] = a[t] - dt - vat 
                    xi[i,t] = xt
                    if vat+vbt>0:
                        cost_shot[i,t] =  p[t]*(vat+vbt)
    return x,d,va,vb,cost_shot

def Aofl(ts,tnz,abar,p):
    mu_c = 100; mu_d =100
    a = np.zeros(len(p))
    a[int(tnz)] = abar
    xt = np.zeros(len(p))
    dt = np.zeros(len(p))
    vat = np.zeros(len(p))
    vbt = np.zeros(len(p))
    try:
        p_min = min(p[int(ts):int(tnz+1)])
    except:
        p_min = 1e8
            
    if p[int(ts)]==p_min:
        dt[int(ts)] = 0
        vat[int(ts)] = a[int(ts)]
        vbt[int(ts)] = min(max(abar-0,0),mu_c)
        xt[int(ts)] = 0 + vbt[int(ts)] - dt[int(ts)]
    else:
        dt[int(ts)] = min(a[int(ts)],mu_d,0)
        vat[int(ts)] = a[int(ts)]-dt[int(ts)]
        vbt[int(ts)] = 0
        xt[int(ts)] = 0 + vbt[int(ts)] - dt[int(ts)]
    
    for t in range(int(ts+1),int(tnz)):
        if p[t] == p_min:
            dt[t] = 0
            vat[t] = a[t]
            vbt[t] = min(max(abar-xt[t-1],0),mu_c)
            xt[t] = xt[t-1] + vbt[t] - dt[t]
        else:
            dt[t] = min(a[t],mu_d,xt[t-1])
            vat[t] = a[t]-dt[t]
            vbt[t] = 0
            xt[t] = xt[t-1] + vbt[t] - dt[t]
            
    dt[int(tnz)] = min(a[int(tnz)],mu_d,xt[int(tnz-1)])
    vat[int(tnz)] = a[int(tnz)]-dt[int(tnz)]
    vbt[int(tnz)] = 0
    xt[int(tnz)] = xt[int(tnz-1)] + vbt[int(tnz)] - dt[int(tnz)]
    
    return xt,dt,vat,vbt

def Aofl_hat(ao,po,B):
    a = ao.copy() ; p = po.copy()
    x = np.zeros(int(len(a)))
    d = np.zeros(int(len(a)))
    va = np.zeros(int(len(a)))
    vb = np.zeros(int(len(a)))
    x0 = 0; x[0] = 0
    abar, ts, tnz = isDecompose(a,B,x0)
    cost_shot = np.zeros(len(abar))
    for i in range(len(abar)):
        xt,dt,vat,vbt = Aofl(ts[i],tnz[i],abar[i],p)
        for t in range(int(ts[i]),int(tnz[i]+1)):
            x[t] = x[t] + xt[t]
            d[t] = d[t] + dt[t]
            va[t] = va[t] + vat[t]
            vb[t] = vb[t] + vbt[t]
            a[t] = a[t] - dt[t] - vat[t]
            if vat[t]+vbt[t]>0:
                cost_shot[i] = cost_shot[i] + p[t]*(vat[t]+vbt[t])
    return x,d,va,vb,cost_shot

###############################################################################
def Norm_pdfx(x):
    return x

def Atheta_gmm(clf,T):
    k,por,mean,std = len(clf.weights_),clf.weights_,clf.means_,clf.covariances_
    t = int(T-2); theta = np.zeros(int(T))
    trunc = 0; re = 0;
    for i in range(k):
        trunc = trunc + por[i]*(1-norm.cdf(0,loc = mean[i], scale = np.sqrt(std[i])))
        re = re + por[i]*norm.expect(Norm_pdfx,loc = mean[i],scale = np.sqrt(std[i]),lb = 0, ub=np.inf)
    theta[int(T-2)] = re/trunc
    while(t>0):
        t = t-1; re1 = 0; re2 = 0; trunc = 0
        for i in range(k):
            trunc = trunc + por[i]*(1-norm.cdf(0,loc = mean[i], scale = np.sqrt(std[i])))
            re1 = re1 + por[i]*norm.expect(Norm_pdfx,loc = mean[i],scale = np.sqrt(std[i]),lb = 0,ub= theta[t+1])
            re2 = re2 + por[i]*theta[t+1] * (1-norm.cdf(theta[t+1],loc = mean[i], scale = np.sqrt(std[i])))
        theta[t] = (re1+re2)/trunc
    return theta

def Aour_hat_gmm(ao,po,B,clf):
    a = ao.copy(); p = po.copy()
    x = np.zeros(int(len(a)))
    d = np.zeros(int(len(a)))
    va = np.zeros(int(len(a)))
    vb = np.zeros(int(len(a)))
    x0 = 0; 
    abar, ts, tnz = isDecompose(a,B,x0)
    xi = np.zeros((len(abar),len(a)))
    theta = np.zeros((len(abar),len(a)))
    cost_shot = np.zeros((len(abar),len(a)))
    
    for i in range(len(abar)):
        theta[i,int(ts[i]):int(tnz[i]+1)] = Atheta_gmm(clf,tnz[i]-ts[i]+1)
    
    for t in range(len(a)):
        for i in range(len(abar)):
            if t>=ts[i] and t<=tnz[i]:
                if t==ts[i]:
                    xt,dt,vat,vbt = Athb_ld(theta[i,t],ts[i],tnz[i],abar[i],p[t],
                                            t,0)
                    x[t] = x[t] + xt
                    d[t] = d[t] + dt
                    va[t] = va[t] + vat
                    vb[t] = vb[t] + vbt
                    a[t] = a[t] - dt - vat 
                    xi[i,t] = xt
                    if vat+vbt>0:
                        cost_shot[i,t] =  p[t]*(vat+vbt)
                else:
                    xt,dt,vat,vbt = Athb_ld(theta[i,t],ts[i],tnz[i],abar[i],p[t],
                                            t,xi[i,t-1])
                    x[t] = x[t] + xt
                    d[t] = d[t] + dt
                    va[t] = va[t] + vat
                    vb[t] = vb[t] + vbt
                    a[t] = a[t] - dt - vat 
                    xi[i,t] = xt
                    if vat+vbt>0:
                        cost_shot[i,t] =  p[t]*(vat+vbt)
    return x,d,va,vb,cost_shot

def Aour_hat_gmm_update(ao,po,B):
    a = ao.copy() ; p = po.copy()
    x = np.zeros(int(len(a)))
    d = np.zeros(int(len(a)))
    va = np.zeros(int(len(a)))
    vb = np.zeros(int(len(a)))
    x0 = 0;
    abar, ts, tnz = isDecompose(a,B,x0)
    xi = np.zeros((len(abar),len(a)))
    cost_shot = np.zeros((len(abar),len(a)))
    for t in range(len(a)):
        try:
           clf = estimate_gmm(p[1:t])
           Theta = Atheta_gmm(clf,len(a))
           print(Theta)
        except:
           continue
        for i in range(len(abar)):
            if t>=ts[i] and t<=tnz[i]:
                theta = np.zeros(len(a))
                theta[int(ts[i]):int(tnz[i]+1)] = Theta[int(len(a)-(tnz[i]-ts[i]+1))::]
                if t==ts[i]:
                    xt,dt,vat,vbt = Athb_ld(theta[t],ts[i],tnz[i],abar[i],p[t],
                                            t,0)
                    x[t] = x[t] + xt
                    d[t] = d[t] + dt
                    va[t] = va[t] + vat
                    vb[t] = vb[t] + vbt
                    a[t] = a[t] - dt - vat 
                    xi[i,t] = xt
                    if vat+vbt>0:
                        cost_shot[i,t] =  p[t]*(vat+vbt)
                else:
                    xt,dt,vat,vbt = Athb_ld(theta[t],ts[i],tnz[i],abar[i],p[t],
                                            t,xi[i,t-1])
                    x[t] = x[t] + xt
                    d[t] = d[t] + dt
                    va[t] = va[t] + vat
                    vb[t] = vb[t] + vbt
                    a[t] = a[t] - dt - vat 
                    xi[i,t] = xt
                    if vat+vbt>0:
                        cost_shot[i,t] =  p[t]*(vat+vbt)
    return x,d,va,vb,cost_shot


def estimate_gmm(X):
    X = X.reshape(-1,1)
    bic = []; lowest_bic = np.infty;
    n_components_range = range(3, 4)
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

def sample_gaussian(mean, covar, covariance_type='diag', n_samples=1,
                    random_state=None):
    n_dim = len(mean)
    rand = np.random.randn(n_dim, n_samples)
    if n_samples == 1:
        rand.shape = (n_dim,)

    if covariance_type == 'spherical':
        rand *= np.sqrt(covar)
    elif covariance_type == 'diag':
        rand = np.dot(np.diag(np.sqrt(covar)), rand)
    else:
        s, U = linalg.eigh(covar)
        s.clip(0, out=s)        # get rid of tiny negatives
        np.sqrt(s, out=s)
        U *= s
        rand = np.dot(U, rand)
    return (rand.T + mean).T

def sample(clf, n_samples=1, random_state=None):

    weight_cdf = np.cumsum(clf.weights_)

    X = np.empty((n_samples, clf.means_.shape[1]))
    rand = np.random.rand(n_samples)
    # decide which component to use for each sample
    comps = weight_cdf.searchsorted(rand)
    print(comps)
    # for each component, generate all needed samples
    for comp in range(clf.n_components):
        # occurrences of current component in X
        comp_in_X = (comp == comps)
        # number of those occurrences
        num_comp_in_X = comp_in_X.sum()
        if num_comp_in_X > 0:
            if clf.covariance_type == 'tied':
                cv = clf.covars_
            elif clf.covariance_type == 'spherical':
                cv = clf.covariances_[comp]
            else:
                cv = clf.covars_[comp]
            X[comp_in_X] = sample_gaussian(
                clf.means_[comp], cv, clf.covariance_type,
                num_comp_in_X, random_state=random_state).T
    return X

###############################################################################
def gnorm(x):
    no = 0
    for i in range(k):
        no = no + por[i]*norm.cdf(x,mean[i],np.sqrt(std[i]))
    return (1-no)/trunc*2*x
    


df = pd.read_csv('RealData.csv')
A = df['Load Data']; 
P = df['Price Data']; 
clf = estimate_gmm(P.values)
N = 50; ITER = 100

trunctry = 11
k,por,mean,std = len(clf.weights_),clf.weights_,clf.means_,clf.covariances_
regret = np.zeros(N-2)
for T in range(2,N):
    theta = Atheta_gmm(clf,T+1)
    betas = 0; alphas = 0;
    
    for x in range(T-1):
        trunc = 0; pdf0 = 0; pdfx = 0;
        for i in range(k):
            trunc = trunc + por[i]*(1-norm.cdf(trunctry,loc = mean[i], scale = np.sqrt(std[i])))
            pdf0 = pdf0 + por[i]*norm.cdf(trunctry,loc = mean[i], scale = np.sqrt(std[i]))
            pdfx = pdfx + por[i]*norm.pdf(theta[-(i+1)],loc = mean[i], scale = np.sqrt(std[i]))
        betas = betas + min(pdfx/trunc,pdf0/trunc)
    
    for i in range(k):
        trunc = trunc + por[i]*(1-norm.cdf(trunctry,loc = mean[i], scale = np.sqrt(std[i])))
        alphas = alphas+por[i]*norm.expect(gnorm,loc=mean[i],scale=np.sqrt(std[i]),lb=trunctry,ub=10000000)
    alphas = alphas/trunc
    
    regret[T-2] = 2/(betas)-T*pow(alphas/(2*theta[T-2]),T-1)*theta[T-2]

regret_uniform = np.zeros((N,ITER))
for n in range(1,N+1):
    for iternum in range(ITER):
        p = sample(clf, n_samples=N, random_state=None)
        p = list(p)
        o = min(p)
        theta = Atheta_gmm(clf,n)
        for t in range(n):
            if p[t]<= theta[t]:
                re= (p[t]-o)/o
                break
            elif t == n-1:
                re = (p[t]-o)/o
                break
        regret_uniform[n-1,iternum] = re


x = list(range(1,N))
min_ser = [np.percentile(i, 5) for i in regret_uniform]
max_ser = [np.percentile(i, 95) for i in regret_uniform]
mean_ser = [np.percentile(i, 50) for i in regret_uniform]
sevn_ser = [np.percentile(i, 75) for i in regret_uniform]
twon_ser = [np.percentile(i, 25) for i in regret_uniform]

fig, axs = plt.subplots()
axs.fill_between(x[1:], min_ser[1:49], max_ser[1:49], color = 'lightslategrey',alpha=0.2, label = "Percentile 5%-95%" )
axs.plot(x[1:], mean_ser[1:49], linewidth = 3,color = 'wheat', label = "Percentile 50%")
axs.plot(x[1:], sevn_ser[1:49],dashes = [3,2], linewidth = 3, color = 'darkseagreen',label = "Percentile 75%")
axs.plot(x[1:], twon_ser[1:49],dashes = [0.5,1], linewidth = 3, color = 'lightsalmon',label = "Percentile 25%")

#axs.plot(x[3:],regret[3:], color = 'r', label='$Regret Bound$')

axs.legend(fontsize=13)
axs.set_xlabel('Time/h',fontsize=15)
axs.set_ylabel("$\gamma$",fontsize=15)
axs.tick_params(labelsize=13)
plt.savefig("shot_gamma.pdf")
###############################################################################