# -*- coding: utf-8 -*-
"""
Created on Fri May 29 00:37:05 2020

@author: jmwu
"""

import math
from scipy.stats import norm
import numpy as np
from scipy import linalg
from sklearn import mixture
from scipy.optimize import linprog
from RL_brain import QLearningTable
import warnings
warnings.filterwarnings("ignore")

class clf_par(object):
    def __init__(self,n_weights,weights,means,covariances):
        self.n_weights = n_weights
        self.weights = weights
        self.means = means
        self.covariances = covariances

class BFramework:
    def __init__(self, XIN, FUL, LEVEL, B, W):
        self.XIN = XIN
        self.FUL = FUL
        self.LEVEL = LEVEL
        self.B = B
        self.W = W

    def stepto(self, action,observation,step,p,pbar,d):
        length = len(p)
        pbar = (1-self.XIN)*pbar+self.XIN*observation[0]
        if action == 0:
            reward = 0
            sl = observation[2]
            ch = 0
            dh = 0
            gh = observation[1]
            stepcost = observation[0]*observation[1]
        elif action == 1:
            reward = (observation[0]-pbar)*min(observation[2],observation[1])
            sl = observation[2]-min(observation[2],observation[1])
            ch = 0
            dh = min(observation[2],observation[1])
            gh = observation[1]-min(observation[2],observation[1])
            stepcost = observation[0]*(observation[1]-min(observation[2],observation[1]))
        elif action == 2:
            reward = (pbar-observation[0])*int((self.B-observation[2])*self.FUL)
            sl = observation[2]+int((self.B-observation[2])*self.FUL)
            ch = int((self.B-observation[2])*self.FUL)
            dh = 0
            gh = observation[1]
            stepcost = observation[0]*(observation[1]+int((self.B-observation[2])*self.FUL))
        
        if step == length-1:
            done = True
            observation_ =  'terminal'
            return observation_, reward, done, pbar, stepcost, sl, ch, dh, gh
        else:
            done = False
            observation_ =  np.array([p[step+1],d[step+1],sl])
            return observation_, reward, done, pbar, stepcost, sl, ch, dh, gh
    
    def train(self, RL,d,p):
        for episode in range(20):
            s = 0; step = 0; pbar = p[0]
            observation = np.array([p[0],d[0],s])
            while True:
                temp_ob = observation.copy()/self.LEVEL; temp_ob = temp_ob.astype(int);
                action = RL.choose_action(str(temp_ob))
                observation_, reward, done, pbar, stepcost,sl, ch, dh, gh = self.stepto(action,observation,step,p,pbar,d)
                if observation_== 'terminal':
                    RL.learn(str(temp_ob), action, reward, observation_)
                    print('terminal episode '+str(episode))
                    break
                else:
                    temp_ob_ = observation_.copy()/self.LEVEL; temp_ob_ = temp_ob_.astype(int);
                    RL.learn(str(temp_ob), action, reward, str(temp_ob_))
                observation = observation_
                step = step + 1
                if step>=(len(d)):
                    break
        return RL
                
    ###########################################################################
    def solvelp(self, pw,dw,pbar,x0):
        #####objective function########
        c = np.r_[-pw,pw,np.zeros(len(pw))]
        #####constraints for capacity##
        for i in range(len(pw)):
            a1 = np.zeros(len(pw))
            a2 = np.zeros(len(pw))
            a3 = np.zeros(len(pw))
            a1[i] = -1
            a2[i] = 1
            if i == 0:
                a3[i] = -1
            else:
                a3[i] = -1
                a3[i-1] = 1
            a = np.r_[np.transpose(a1),np.transpose(a2),np.transpose(a3)]
            if i==0:
                A_e =  np.array([a])
            else:
                A_e =  np.r_[A_e,[a]]
        b_e = np.zeros(len(pw)); b_e[0] = -x0
        #####constraints for upper#####
        for i in range(len(pw)):
            a1 = np.zeros(len(pw))
            a2 = np.zeros(len(pw))
            a3 = np.zeros(len(pw))
            a3[i] = 1
            a = np.r_[np.transpose(a1),np.transpose(a2),np.transpose(a3)]
            if i==0:
                A_u1 =  np.array([a])
            else:
                A_u1 =  np.r_[A_u1,[a]]
        b_u1 = np.zeros(len(pw))+self.B
        
        for i in range(len(pw)):
            a1 = np.zeros(len(pw))
            a2 = np.zeros(len(pw))
            a3 = np.zeros(len(pw))
            a1[i] = 1
            a = np.r_[np.transpose(a1),np.transpose(a2),np.transpose(a3)]
            if i==0:
                A_u2 =  np.array([a])
            else:
                A_u2 =  np.r_[A_u2,[a]]
        b_u2 = np.zeros(len(pw))+self.B
        
        for i in range(len(pw)):
            a1 = np.zeros(len(pw))
            a2 = np.zeros(len(pw))
            a3 = np.zeros(len(pw))
            a2[i] = 1
            a = np.r_[np.transpose(a1),np.transpose(a2),np.transpose(a3)]
            if i==0:
                A_u3 =  np.array([a])
            else:
                A_u3 =  np.r_[A_u3,[a]]
        b_u3 = np.zeros(len(pw))+self.B
        
        for i in range(len(pw)):
            a1 = np.zeros(len(pw))
            a2 = np.zeros(len(pw))
            a3 = np.zeros(len(pw))
            a1[i] = 1
            a = np.r_[np.transpose(a1),np.transpose(a2),np.transpose(a3)]
            if i==0:
                A_u4 =  np.array([a])
            else:
                A_u4 =  np.r_[A_u4,[a]]
        b_u4 = np.zeros(len(pw))+dw
        
        A_u = np.r_[A_u1,A_u2]
        A_u = np.r_[A_u,A_u3]
        A_u = np.r_[A_u,A_u4]
        
        b_u = np.r_[b_u1,b_u2]
        b_u = np.r_[b_u,b_u3]
        b_u = np.r_[b_u,b_u4]
        ################################
        res = linprog(c, A_ub=A_u, b_ub=b_u, A_eq=A_e, b_eq = b_e)
        return res.x
    
    def A_mcp(self, p,d,mu):
        T = len(p)
        cost = np.zeros(T)
        dc = np.zeros(T)
        cc = np.zeros(T)
        xc = np.zeros(T)
        gc = np.zeros(T)
        xcc = 0
        for t in range(T):
            if t+self.W<=T:
                pw = np.r_[np.array([p[t]]),np.zeros(self.W-1)+mu]
                dw = d[t:t+self.W]
                x = self.solvelp(pw,dw,mu,xcc)
                dc[t] = x[0]
                cc[t] = x[self.W]
                xc[t] = x[2*self.W]
                gc[t] = d[t]-dc[t]
                xcc = xc[t]
                cost[t] = p[t]*(cc[t]+d[t]-dc[t])
            else: 
                w = T-t
                pw = np.r_[np.array([p[t]]),np.zeros(w-1)+mu]
                dw = d[t:t+w]
                x = self.solvelp(pw,dw,mu,xcc)
                dc[t] = x[0]
                cc[t] = x[w]
                xc[t] = x[2*w]
                gc[t] = d[t]-dc[t]
                xcc = xc[t]
                cost[t] = p[t]*(cc[t]+d[t]-dc[t])
        return xc,dc,gc,cc,cost
    
    ###########################################################################
    def isDecompose(self,at,x0):
        Acc_sum = 0
        Acc = np.zeros(len(at))
        for i in range(len(at)):
            Acc[i] = Acc_sum + at[i]
            Acc_sum = Acc_sum + at[i]
        AccB = Acc + self.B
        
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
    
    
    def Athb(self, theta,BB,t,at,pt,xt0,tnz):
        mu_c = 100000; mu_d =100000
        if t == tnz:
            dt =  min(at,mu_d,xt0)
            vat = at - dt
            vbt = 0
            xt = xt0+vbt-dt
            return xt,dt,vat,vbt
        if pt<=theta:
            dt = 0
            vat = at
            vbt = min(max(BB-xt0,0),mu_c)
        else:
            dt =  min(at,mu_d,xt0)
            vat = at - dt
            vbt = 0
        xt = xt0+vbt-dt
        return xt,dt,vat,vbt
                
    def Athb_ld(self,theta,ts,tnz,abar,pt,t,xt0):
        if t == tnz:
            abart = abar
        else:
            abart = 0
        return self.Athb(theta,abar,t,abart,pt,xt0,tnz)
    
    def Athb_hat(self, ao,po,theta):
        a = ao.copy() ; p = po.copy()
        x = np.zeros(int(len(a)))
        d = np.zeros(int(len(a)))
        va = np.zeros(int(len(a)))
        vb = np.zeros(int(len(a)))
        x0 = 0;
        abar, ts, tnz = self.isDecompose(a,x0)
        xi = np.zeros((len(abar),len(a)))
        cost_shot = np.zeros((len(abar),len(a)))
        for t in range(len(a)):
            for i in range(len(abar)):
                if t>=ts[i] and t<=tnz[i]:
                    if t==ts[i]:
                        xt,dt,vat,vbt = self.Athb_ld(theta,ts[i],tnz[i],abar[i],p[t],
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
                        xt,dt,vat,vbt = self.Athb_ld(theta,ts[i],tnz[i],abar[i],p[t],
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
    
    def Aofl(self,ts,tnz,abar,p):
        mu_c = 100000; mu_d =100000
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
    
    def Aofl_hat(self,ao,po):
        a = ao.copy() ; p = po.copy()
        x = np.zeros(int(len(a)))
        d = np.zeros(int(len(a)))
        va = np.zeros(int(len(a)))
        vb = np.zeros(int(len(a)))
        x0 = 0; x[0] = 0
        abar, ts, tnz = self.isDecompose(a,x0)
        cost_shot = np.zeros(len(a))
        for i in range(len(abar)):
            xt,dt,vat,vbt = self.Aofl(ts[i],tnz[i],abar[i],p)
            for t in range(int(ts[i]),int(tnz[i]+1)):
                x[t] = x[t] + xt[t]
                d[t] = d[t] + dt[t]
                va[t] = va[t] + vat[t]
                vb[t] = vb[t] + vbt[t]
                a[t] = a[t] - dt[t] - vat[t]
                if vat[t]+vbt[t]>0:
                    cost_shot[t] = cost_shot[t] + p[t]*(vat[t]+vbt[t])
        return x,d,va,vb,cost_shot
    
    ###########################################################################
    def Norm_pdfx(self,x):
        return x
    
    def estimate_gmm(self,X):
        X = X.reshape(-1,1)
        bic = []; lowest_bic = np.infty;
        n_components_range = range(1, 4)
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
    
    def sample_gaussian(self,mean, covar, covariance_type='diag', n_samples=1,
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
            s.clip(0, out=s)        
            np.sqrt(s, out=s)
            U *= s
            rand = np.dot(U, rand)
        return (rand.T + mean).T
    
    def sample(self,clf, n_samples=1, random_state=None):
    
        weight_cdf = np.cumsum(clf.weights_)
    
        X = np.empty((n_samples, clf.means_.shape[1]))
        rand = np.random.rand(n_samples)
        comps = weight_cdf.searchsorted(rand)
        for comp in range(clf.n_components):
            comp_in_X = (comp == comps)
            num_comp_in_X = comp_in_X.sum()
            if num_comp_in_X > 0:
                if clf.covariance_type == 'tied':
                    cv = clf.covars_
                elif clf.covariance_type == 'spherical':
                    cv = clf.covariances_[comp]
                else:
                    cv = clf.covars_[comp]
                X[comp_in_X] = self.sample_gaussian(
                    clf.means_[comp], cv, clf.covariance_type,
                    num_comp_in_X, random_state=random_state).T
        return X
    
    def Atheta_gmm(self,clf,T):
        k,por,mean,std = len(clf.weights_),clf.weights_,clf.means_,clf.covariances_
        t = int(T-2); theta = np.zeros(int(T))
        trunc = 0; re = 0;
        for i in range(k):
            trunc = trunc + por[i]*(1-norm.cdf(0,loc = mean[i], scale = np.sqrt(std[i])))
            re = re + por[i]*norm.expect(self.Norm_pdfx,loc = mean[i],scale = np.sqrt(std[i]),lb = 0, ub= np.inf)
        theta[int(T-2)] = re/trunc
        while(t>0):
            t = t-1; re1 = 0; re2 = 0; trunc = 0
            for i in range(k):
                trunc = trunc + por[i]*(1-norm.cdf(0,loc = mean[i], scale = np.sqrt(std[i])))
                re1 = re1 + por[i]*norm.expect(self.Norm_pdfx,loc = mean[i],scale = np.sqrt(std[i]),lb = 0,ub= theta[t+1])
                re2 = re2 + por[i]*theta[t+1] * (1-norm.cdf(theta[t+1],loc = mean[i], scale = np.sqrt(std[i])))
            theta[t] = (re1+re2)/trunc
        return theta
    
    def Aour_hat_gmm(self,ao,po,clf):
        a = ao.copy(); p = po.copy()
        x = np.zeros(int(len(a)))
        d = np.zeros(int(len(a)))
        va = np.zeros(int(len(a)))
        vb = np.zeros(int(len(a)))
        x0 = 0; 
        abar, ts, tnz = self.isDecompose(a,x0)
        xi = np.zeros((len(abar),len(a)))
        theta = np.zeros((len(abar),len(a)))
        cost_shot = np.zeros((len(abar),len(a)))
        
        Theta = self.Atheta_gmm(clf,len(a))
        
        for t in range(len(a)):
            for i in range(len(abar)):
                if t>=ts[i] and t<=tnz[i]:
                    theta = Theta[len(a)-int(tnz[i]-t)-1]
                    if t==ts[i]:
                        xt,dt,vat,vbt = self.Athb_ld(theta,ts[i],tnz[i],abar[i],p[t],
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
                        xt,dt,vat,vbt = self.Athb_ld(theta,ts[i],tnz[i],abar[i],p[t],
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

###############################################################################
    def estimate_gmm_cor(self,X):
        clfp = {}
        X = X.reshape(-1,1)
        for t in range(24):
            x = X[t:len(X):24]
            bic = []; lowest_bic = np.infty;
            n_components_range = range(1, 4)
            cv_types = ['spherical']
            for cv_type in cv_types:
                for n_components in n_components_range:
                    gmm = mixture.GaussianMixture(n_components=n_components,
                                                  covariance_type=cv_type)
                    gmm.fit(x)
                    bic.append(gmm.bic(x))
                    if bic[-1] < lowest_bic:
                        lowest_bic = bic[-1]
                        best_gmm = gmm
            clfp[t] = clf_par(len(best_gmm.weights_),best_gmm.weights_,best_gmm.means_,best_gmm.covariances_)
        return clfp
    
    def estimate_peak(self,X):
        thres = np.mean(X)
        peak = np.zeros(24)
        for t in range(24):
            x = X[t:len(X):24]
            if np.percentile(x,80) > thres:
                peak[t] = 1
            else:
                peak[t] = 0
        return peak
            
    def estimate_gmm_cor_p(self,X,peak):
        clfp = {}
        X = X.reshape(-1,1)
    ###########################################################################
        for t in range(24):
            if peak[t]==1:
                x1 = X[t:len(X):24]
                break
        
        for t in range(24):
            if peak[t]==1:
                x = X[t:len(X):24]
                x1 = np.vstack((x1,x))
        
        bic = []; lowest_bic = np.infty;
        n_components_range = range(1, 4)
        cv_types = ['spherical']
        for cv_type in cv_types:
            for n_components in n_components_range:
                gmm = mixture.GaussianMixture(n_components=n_components,
                                              covariance_type=cv_type)
                gmm.fit(x1)
                bic.append(gmm.bic(x1))
                if bic[-1] < lowest_bic:
                    lowest_bic = bic[-1]
                    best_gmm = gmm    
    
        for t in range(24):
            if peak[t]==1:
                clfp[t] = clf_par(len(best_gmm.weights_),best_gmm.weights_,best_gmm.means_,best_gmm.covariances_)
    ###########################################################################
        for t in range(24):
            if peak[t]==0:
                x0 = X[t:len(X):24]
                break
                    
        for t in range(24):
            if peak[t]==0:
                x = X[t:len(X):24]
                x0 = np.vstack((x0,x))
                
        bic = []; lowest_bic = np.infty;
        n_components_range = range(1, 4)
        cv_types = ['spherical']
        for cv_type in cv_types:
            for n_components in n_components_range:
                gmm = mixture.GaussianMixture(n_components=n_components,
                                              covariance_type=cv_type)
                gmm.fit(x0)
                bic.append(gmm.bic(x0))
                if bic[-1] < lowest_bic:
                    lowest_bic = bic[-1]
                    best_gmm = gmm
        clfp[t] = clf_par(len(best_gmm.weights_),best_gmm.weights_,best_gmm.means_,best_gmm.covariances_)        
                
        for t in range(24):
            if peak[t]==0:
                clfp[t] = clf_par(len(best_gmm.weights_),best_gmm.weights_,best_gmm.means_,best_gmm.covariances_)
                
        return clfp
    
    def Atheta_gmm_cor(self,clfp,edt,T):
        clf = clfp[edt]
        k,por,mean,std = clf.n_weights,clf.weights,clf.means,clf.covariances
        t = int(T-2); theta = np.zeros(int(T))
        trunc = 0; re = 0;
        for i in range(k):
            trunc = trunc + por[i]*(1-norm.cdf(0,loc = mean[i], scale = np.sqrt(std[i])))
            re = re + por[i]*norm.expect(self.Norm_pdfx,loc = mean[i],scale = np.sqrt(std[i]),lb = 0, ub= np.inf)
        theta[int(T-2)] = re/trunc
        
        est = edt - 1
        while(t>0):
            t = t-1; re1 = 0; re2 = 0; trunc = 0
            if(est<0):
                est = 23
            clf = clfp[est]
            k,por,mean,std = clf.n_weights,clf.weights,clf.means,clf.covariances
            for i in range(k):
                trunc = trunc + por[i]*(1-norm.cdf(0,loc = mean[i], scale = np.sqrt(std[i])))
                re1 = re1 + por[i]*norm.expect(self.Norm_pdfx,loc = mean[i],scale = np.sqrt(std[i]),lb = 0,ub= theta[t+1])
                re2 = re2 + por[i]*theta[t+1] * (1-norm.cdf(theta[t+1],loc = mean[i], scale = np.sqrt(std[i])))
            theta[t] = (re1+re2)/trunc
        return theta
    
    def Aour_hat_gmm_cor(self,ao,po,clfp):
        a = ao.copy(); p = po.copy()
        x = np.zeros(int(len(a)))
        d = np.zeros(int(len(a)))
        va = np.zeros(int(len(a)))
        vb = np.zeros(int(len(a)))
        x0 = 0; 
        abar, ts, tnz = self.isDecompose(a,x0)
        xi = np.zeros((len(abar),len(a)))
        theta = np.zeros((len(abar),len(a)))
        cost_shot = np.zeros((len(abar),len(a)))
        
        Theta = np.zeros((24,len(a)))
        for i in range(24):
            Theta[i,:] = self.Atheta_gmm_cor(clfp,i,len(a))
        
        for t in range(len(a)):
            for i in range(len(abar)):
                if t>=ts[i] and t<=tnz[i]:
                    edt = (tnz[i]+1)%24-1
                    if edt<0:
                        edt = 23
                    theta = Theta[int(edt),len(a)-int(tnz[i]-t)-1]
                    if t==ts[i]: 
                        xt,dt,vat,vbt = self.Athb_ld(theta,ts[i],tnz[i],abar[i],p[t],
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
                        xt,dt,vat,vbt = self.Athb_ld(theta,ts[i],tnz[i],abar[i],p[t],
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
    
    def Aour_hat_gmm_cor_p(self,ao,po,clfp):
        a = ao.copy(); p = po.copy()
        x = np.zeros(int(len(a)))
        d = np.zeros(int(len(a)))
        va = np.zeros(int(len(a)))
        vb = np.zeros(int(len(a)))
        x0 = 0; 
        abar, ts, tnz = self.isDecompose(a,x0)
        xi = np.zeros((len(abar),len(a)))
        theta = np.zeros((len(abar),len(a)))
        cost_shot = np.zeros((len(abar),len(a)))
        
        Theta = np.zeros((24,len(a)))
        for i in range(24):
            Theta[i,:] = self.Atheta_gmm_cor(clfp,i,len(a))
        
        for t in range(len(a)):
            for i in range(len(abar)):
                if t>=ts[i] and t<=tnz[i]:
                    edt = (tnz[i]+1)%24-1
                    if edt<0:
                        edt = 23
                    theta = Theta[int(edt),len(a)-int(tnz[i]-t)-1]
                    if t==ts[i]: 
                        xt,dt,vat,vbt = self.Athb_ld(theta,ts[i],tnz[i],abar[i],p[t],
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
                        xt,dt,vat,vbt = self.Athb_ld(theta,ts[i],tnz[i],abar[i],p[t],
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
    
    
    def general_performance(self,D,P):
        ###############################pre data################################
        fold_i = 730; fold_n = 2 
        socofl = np.zeros((fold_n+1,int(fold_i+1),4))
        socour = np.zeros((fold_n+1,int(fold_i+1),4))
        socrl = np.zeros((fold_n+1,int(fold_i+1),4))
        socmpc = np.zeros((fold_n+1,int(fold_i+1),4))
        socthb = np.zeros((fold_n+1,int(fold_i+1),4))
        socnos = np.zeros((fold_n+1,int(fold_i+1),4))
        ###############################split data##############################
        for f in range(fold_n):
            ###################################################################
            ########################train######################################
            d = D[f*fold_i:(f+2)*fold_i]; d = np.r_[0,d]; d = d.round().astype(int);
            p = P[f*fold_i:(f+2)*fold_i]; p = np.r_[np.mean(p),p];
            theta = math.sqrt(max(p)*min(p))
            clf = self.estimate_gmm(p); ex = 0;
            k,por,mean,std = len(clf.weights_),clf.weights_,clf.means_,clf.covariances_
            for i in range(k):
                ex = ex + por[i]*norm.expect(self.Norm_pdfx,loc = mean[i],scale = np.sqrt(std[i]),lb = 0, ub= np.inf)
            RL = QLearningTable(actions=list(range(3)))
            RL = self.train(RL,d,p)
            print('Training Done')
            ###################################################################
            ########################test#######################################
            d = D[(f+2)*fold_i:(f+3)*fold_i]; d = np.r_[0,d]; d = d.round().astype(int);
            p = P[(f+2)*fold_i:(f+3)*fold_i]; p = np.r_[np.mean(p),p];
            #############################DETA##################################
            xo,do,vao,vbo,cost_o = self.Aour_hat_gmm(d,p,clf)
            ###########################MPC#####################################
            xm,dm,vam,vbm,cost_ep = self.A_mcp(p,d,ex)
            #############################OFL###################################
            xf,df,vaf,vbf,cost_ofl = self.Aofl_hat(d,p)
            #############################THB###################################
            xh,dh,vah,vbh,cost_h = self.Athb_hat(d,p,theta)
            #############################RL####################################
            s = 0; step = 0; cost_r = 0; pbar = p[0]
            observation = np.array([p[0],d[0],s])
            while True:
                temp_ob = observation.copy()/self.LEVEL; temp_ob = temp_ob.astype(int);
                action = RL.choose_action(str(temp_ob))
                observation_, reward, done, pbar, stepcost,sd, cd, dd, gd = self.stepto(action,observation,step,p,pbar,d)
                if observation_== 'terminal':
                    cost_r = cost_r + stepcost
                    socrl[f,step,0] = sd
                    socrl[f,step,1] = dd
                    socrl[f,step,2] = gd
                    socrl[f,step,3] = cd
                    break
                else:
                    cost_r = cost_r + stepcost
                    socrl[f,step,0] = sd
                    socrl[f,step,1] = dd
                    socrl[f,step,2] = gd
                    socrl[f,step,3] = cd
                observation = observation_
                step = step + 1
                if step>=fold_i+1:
                    break
            ##########################save result##############################
            socmpc[f,:,0] = xm
            socmpc[f,:,1] = dm
            socmpc[f,:,2] = vam
            socmpc[f,:,3] = vbm
            
            socofl[f,:,0] = xf
            socofl[f,:,1] = df
            socofl[f,:,2] = vaf
            socofl[f,:,3] = vbf
            
            socour[f,:,0] = xo
            socour[f,:,1] = do
            socour[f,:,2] = vao
            socour[f,:,3] = vbo
            
            socthb[f,:,0] = xh
            socthb[f,:,1] = dh
            socthb[f,:,2] = vah
            socthb[f,:,3] = vbh
            
            socnos[f,:,0] = np.zeros(fold_i+1)
            socnos[f,:,1] = np.zeros(fold_i+1)
            socnos[f,:,2] = d
            socnos[f,:,3] = np.zeros(fold_i+1)
        ##################compute cost#########################################
        np.save('./data/socour'+str(self.B)+'.npy',socour)
        np.save('./data/socrl'+str(self.B)+'.npy',socrl)
        np.save('./data/socmpc'+str(self.B)+'.npy',socmpc)
        np.save('./data/socnos'+str(self.B)+'.npy',socnos)
        np.save('./data/socthb'+str(self.B)+'.npy',socthb)
        np.save('./data/socofl'+str(self.B)+'.npy',socofl)
    
        return socour,socrl,socmpc,socnos,socthb,socofl
    
    def general_ratio(self,D,P):
        ###############################pre data################################
        fold_i = 730; fold_n = 12-2; interval = 5
        costour = np.zeros((int(fold_i/interval),fold_n))
        costrl = np.zeros((int(fold_i/interval),fold_n))
        costmpc = np.zeros((int(fold_i/interval),fold_n))
        costthb = np.zeros((int(fold_i/interval),fold_n))
        costnos = np.zeros((int(fold_i/interval),fold_n))
        costofl = np.zeros((int(fold_i/interval),fold_n))
        ###############################split data##############################
        for f in range(fold_n):
            ###################################################################
            ########################train######################################
            d = D[f*fold_i:(f+2)*fold_i]; d = np.r_[0,d]; d = d.round().astype(int);
            p = P[f*fold_i:(f+2)*fold_i]; p = np.r_[np.mean(p),p];
            theta = math.sqrt(max(p)*min(p))
            clf = self.estimate_gmm(p); ex = 0;
            k,por,mean,std = len(clf.weights_),clf.weights_,clf.means_,clf.covariances_
            for i in range(k):
                ex = ex + por[i]*norm.expect(self.Norm_pdfx,loc = mean[i],scale = np.sqrt(std[i]),lb = 0, ub= np.inf)
            RL = QLearningTable(actions=list(range(3)))
            RL = self.train(RL,d,p)
            print('Training Done')
            ###################################################################
            ########################test#######################################
            d = D[(f+2)*fold_i:(f+3)*fold_i]; d = np.r_[0,d]; d = d.round().astype(int);
            p = P[(f+2)*fold_i:(f+3)*fold_i]; p = np.r_[np.mean(p),p];
            for j in range(interval,len(d),interval):
                #############################DETA##############################
                xo,do,vao,vbo,cost_o = self.Aour_hat_gmm(d[:j],p[:j],clf)
                ###########################MPC#################################
                xm,dm,vam,vbm,cost_ep = self.A_mcp(p[:j],d[:j],ex)
                #############################OFL###############################
                xf,df,vaf,vbf,cost_ofl = self.Aofl_hat(d[:j],p[:j])
                #############################THB###############################
                xh,dh,vah,vbh,cost_h = self.Athb_hat(d[:j],p[:j],theta)
                #############################RL################################
                s = 0; step = 0; cost_r = 0; pbar = p[0]
                observation = np.array([p[0],d[0],s])
                while True:
                    temp_ob = observation.copy()/self.LEVEL; temp_ob = temp_ob.astype(int);
                    action = RL.choose_action(str(temp_ob))
                    observation_, reward, done, pbar, stepcost,sl, cd, dd, gd = self.stepto(action,observation,step,p[:j],pbar,d[:j])
                    if observation_== 'terminal':
                        cost_r = cost_r + stepcost
                        break
                    else:
                        cost_r = cost_r + stepcost
                    observation = observation_
                    step = step + 1
                    if step>=j:
                        break
                ##########################save result##########################
                costrl[int(j/interval)-1,f] = cost_r/sum(cost_ofl)
                costour[int(j/interval)-1,f] = sum(sum(cost_o))/sum(cost_ofl)
                costmpc[int(j/interval)-1,f] = sum(cost_ep)/sum(cost_ofl)
                costnos[int(j/interval)-1,f] = sum(np.array(d[:j])*np.array(p[:j]))/sum(cost_ofl)
                costthb[int(j/interval)-1,f] = sum(sum(cost_h))/sum(cost_ofl)
                costofl[int(j/interval)-1,f] = sum(cost_ofl)
                print('###########################################################')
                print(sum(np.array(d[:j])*np.array(p[:j])))
                print(cost_r)
                print(sum(sum(cost_o)))
                print(sum(cost_ep))
                print(sum(sum(cost_h)))
                print(sum(cost_ofl))
        ##################compute cost#########################################
        np.save('./data/costour'+str(self.B)+'.npy',costour)
        np.save('./data/costrl'+str(self.B)+'.npy',costrl)
        np.save('./data/costmpc'+str(self.B)+'.npy',costmpc)
        np.save('./data/costnos'+str(self.B)+'.npy',costnos)
        np.save('./data/costthb'+str(self.B)+'.npy',costthb)
        np.save('./data/costofl'+str(self.B)+'.npy',costofl)
        
        return costour,costrl,costmpc,costnos,costthb,costofl


    def general_performance_sys(self,D,P):
        ###############################pre data################################
        fold_i = 730; fold_n = 12-2
        clf = self.estimate_gmm(P)
        P = self.sample(clf, n_samples=fold_i*12, random_state=None)
        P = P[:,0]
        
        socofl = np.zeros((fold_n+1,int(fold_i+1),4))
        socour = np.zeros((fold_n+1,int(fold_i+1),4))
        socrl = np.zeros((fold_n+1,int(fold_i+1),4))
        socmpc = np.zeros((fold_n+1,int(fold_i+1),4))
        socthb = np.zeros((fold_n+1,int(fold_i+1),4))
        socnos = np.zeros((fold_n+1,int(fold_i+1),4))
        ###############################split data##############################
        for f in range(fold_n):
            ###################################################################
            ########################train######################################
            d = D[f*fold_i:(f+2)*fold_i]; d = np.r_[0,d]; d = d.round().astype(int);
            p = P[f*fold_i:(f+2)*fold_i]; p = np.r_[np.mean(p),p];
            theta = math.sqrt(max(p)*min(p))
            clf = self.estimate_gmm(p); ex = 0;
            k,por,mean,std = len(clf.weights_),clf.weights_,clf.means_,clf.covariances_
            for i in range(k):
                ex = ex + por[i]*norm.expect(self.Norm_pdfx,loc = mean[i],scale = np.sqrt(std[i]),lb = 0, ub= np.inf)
            RL = QLearningTable(actions=list(range(3)))
            RL = self.train(RL,d,p)
            print('Training Done')
            ###################################################################
            ########################test#######################################
            d = D[(f+2)*fold_i:(f+3)*fold_i]; d = np.r_[0,d]; d = d.round().astype(int);
            p = P[(f+2)*fold_i:(f+3)*fold_i]; p = np.r_[np.mean(p),p];
            #############################DETA##################################
            xo,do,vao,vbo,cost_o = self.Aour_hat_gmm(d,p,clf)
            ###########################MPC#####################################
            xm,dm,vam,vbm,cost_ep = self.A_mcp(p,d,ex)
            #############################OFL###################################
            xf,df,vaf,vbf,cost_ofl = self.Aofl_hat(d,p)
            #############################THB###################################
            xh,dh,vah,vbh,cost_h = self.Athb_hat(d,p,theta)
            #############################RL####################################
            s = 0; step = 0; cost_r = 0; pbar = p[0]
            observation = np.array([p[0],d[0],s])
            while True:
                temp_ob = observation.copy()/self.LEVEL; temp_ob = temp_ob.astype(int);
                action = RL.choose_action(str(temp_ob))
                observation_, reward, done, pbar, stepcost,sd, cd, dd, gd = self.stepto(action,observation,step,p,pbar,d)
                if observation_== 'terminal':
                    cost_r = cost_r + stepcost
                    socrl[f,step,0] = sd
                    socrl[f,step,1] = dd
                    socrl[f,step,2] = gd
                    socrl[f,step,3] = cd
                    break
                else:
                    cost_r = cost_r + stepcost
                    socrl[f,step,0] = sd
                    socrl[f,step,1] = dd
                    socrl[f,step,2] = gd
                    socrl[f,step,3] = cd
                observation = observation_
                step = step + 1
                if step>=fold_i+1:
                    break
            ##########################save result##############################
            socmpc[f,:,0] = xm
            socmpc[f,:,1] = dm
            socmpc[f,:,2] = vam
            socmpc[f,:,3] = vbm
            
            socofl[f,:,0] = xf
            socofl[f,:,1] = df
            socofl[f,:,2] = vaf
            socofl[f,:,3] = vbf
            
            socour[f,:,0] = xo
            socour[f,:,1] = do
            socour[f,:,2] = vao
            socour[f,:,3] = vbo
            
            socthb[f,:,0] = xh
            socthb[f,:,1] = dh
            socthb[f,:,2] = vah
            socthb[f,:,3] = vbh
            
            socnos[f,:,0] = np.zeros(fold_i+1)
            socnos[f,:,1] = np.zeros(fold_i+1)
            socnos[f,:,2] = d
            socnos[f,:,3] = np.zeros(fold_i+1)
        ##################compute cost#########################################
        np.save('./data/socour_sys'+str(self.B)+'.npy',socour)
        np.save('./data/socrl_sys'+str(self.B)+'.npy',socrl)
        np.save('./data/socmpc_sys'+str(self.B)+'.npy',socmpc)
        np.save('./data/socnos_sys'+str(self.B)+'.npy',socnos)
        np.save('./data/socthb_sys'+str(self.B)+'.npy',socthb)
        np.save('./data/socofl_sys'+str(self.B)+'.npy',socofl)
        return socour,socrl,socmpc,socnos,socthb,socofl
    
    def general_ratio_sys(self,D,P):
        ###############################pre data################################
        fold_i = 730; fold_n = 12-2; interval = 5
        
        clf = self.estimate_gmm(P)
        P = self.sample(clf, n_samples=fold_i*12, random_state=None)
        P = P[:,0]
        
        costour = np.zeros((int(fold_i/interval),fold_n))
        costrl = np.zeros((int(fold_i/interval),fold_n))
        costmpc = np.zeros((int(fold_i/interval),fold_n))
        costthb = np.zeros((int(fold_i/interval),fold_n))
        costnos = np.zeros((int(fold_i/interval),fold_n))
        costofl = np.zeros((int(fold_i/interval),fold_n))
        ###############################split data##############################
        for f in range(fold_n):
            ###################################################################
            ########################train######################################
            d = D[f*fold_i:(f+2)*fold_i]; d = np.r_[0,d]; d = d.round().astype(int);
            p = P[f*fold_i:(f+2)*fold_i]; p = np.r_[np.mean(p),p];
            theta = math.sqrt(max(p)*min(p))
            clf = self.estimate_gmm(p); ex = 0;
            k,por,mean,std = len(clf.weights_),clf.weights_,clf.means_,clf.covariances_
            for i in range(k):
                ex = ex + por[i]*norm.expect(self.Norm_pdfx,loc = mean[i],scale = np.sqrt(std[i]),lb = 0, ub= np.inf)
            RL = QLearningTable(actions=list(range(3)))
            RL = self.train(RL,d,p)
            print('Training Done')
            ###################################################################
            ########################test#######################################
            d = D[(f+2)*fold_i:(f+3)*fold_i]; d = np.r_[0,d]; d = d.round().astype(int);
            p = P[(f+2)*fold_i:(f+3)*fold_i]; p = np.r_[np.mean(p),p];
            for j in range(interval,len(d),interval):
                #############################DETA##############################
                xo,do,vao,vbo,cost_o = self.Aour_hat_gmm(d[:j],p[:j],clf)
                ###########################MPC#################################
                xm,dm,vam,vbm,cost_ep = self.A_mcp(p[:j],d[:j],ex)
                #############################OFL###############################
                xf,df,vaf,vbf,cost_ofl = self.Aofl_hat(d[:j],p[:j])
                #############################THB###############################
                xh,dh,vah,vbh,cost_h = self.Athb_hat(d[:j],p[:j],theta)
                #############################RL################################
                s = 0; step = 0; cost_r = 0; pbar = p[0]
                observation = np.array([p[0],d[0],s])
                while True:
                    temp_ob = observation.copy()/self.LEVEL; temp_ob = temp_ob.astype(int);
                    action = RL.choose_action(str(temp_ob))
                    observation_, reward, done, pbar, stepcost,sl, cd, dd, gd = self.stepto(action,observation,step,p[:j],pbar,d[:j])
                    if observation_== 'terminal':
                        cost_r = cost_r + stepcost
                        break
                    else:
                        cost_r = cost_r + stepcost
                    observation = observation_
                    step = step + 1
                    if step>=j:
                        break
                ##########################save result##########################
                costrl[int(j/interval)-1,f] = cost_r/sum(cost_ofl)
                costour[int(j/interval)-1,f] = sum(sum(cost_o))/sum(cost_ofl)
                costmpc[int(j/interval)-1,f] = sum(cost_ep)/sum(cost_ofl)
                costnos[int(j/interval)-1,f] = sum(np.array(d[:j])*np.array(p[:j]))/sum(cost_ofl)
                costthb[int(j/interval)-1,f] = sum(sum(cost_h))/sum(cost_ofl)
                costofl[int(j/interval)-1,f] = sum(cost_ofl)
                print('###########################################################')
                print(sum(np.array(d[:j])*np.array(p[:j])))
                print(cost_r)
                print(sum(sum(cost_o)))
                print(sum(cost_ep))
                print(sum(sum(cost_h)))
                print(sum(cost_ofl))
        ##################compute cost#########################################
        np.save('./data/costour_sys'+str(self.B)+'.npy',costour)
        np.save('./data/costrl_sys'+str(self.B)+'.npy',costrl)
        np.save('./data/costmpc_sys'+str(self.B)+'.npy',costmpc)
        np.save('./data/costnos_sys'+str(self.B)+'.npy',costnos)
        np.save('./data/costthb_sys'+str(self.B)+'.npy',costthb)
        np.save('./data/costofl_sys'+str(self.B)+'.npy',costofl)
        return costour,costrl,costmpc,costnos,costthb,costofl
    
    def casestudy_value(self,df):
        month = 12; N = 168; inteval = 5;
        cost_o = np.zeros((int(N/inteval),month))
        cost_c = np.zeros((int(N/inteval),month))
        cost_p = np.zeros((int(N/inteval),month))
        cost_f = np.zeros((int(N/inteval),month))

        for iternum in range(month):
            df_temp = df[df['Month'] == iternum+1]
            P = df_temp['Price Data']
            D = df_temp['demand']

            peak = self.estimate_peak(P[:504].values)
            clfp = self.estimate_gmm_cor_p(P[:504].values,peak)
            clfc = self.estimate_gmm_cor(P[:504].values)
            clfo = self.estimate_gmm(P[:504].values)
            s = 0
            for n in range(inteval,N+1,inteval):
                d = D[504:504+n].values; d = np.r_[0,d]; d = d.round().astype(int);
                p = P[504:504+n].values; p = np.r_[np.mean(p),p];
                xo,do,vao,vbo,cost_po = self.Aour_hat_gmm_cor_p(d,p,clfp)
                xo,do,vao,vbo,cost_co = self.Aour_hat_gmm_cor(d,p,clfc)
                xo,do,vao,vbo,cost_oo = self.Aour_hat_gmm(d,p,clfo)
                xf,do,vaf,vbf,cost_fo = self.Aofl_hat(d,p)
                cost_o[s,iternum] = sum(sum(cost_oo))/sum(cost_fo)
                cost_c[s,iternum] = sum(sum(cost_co))/sum(cost_fo)
                cost_p[s,iternum] = sum(sum(cost_po))/sum(cost_fo)
                cost_f[s,iternum] = sum(cost_fo)
                s = s + 1
        
        np.save('./data/costceta'+str(self.B)+'.npy',cost_c)
        np.save('./data/costpeta'+str(self.B)+'.npy',cost_p)
        np.save('./data/costdeta'+str(self.B)+'.npy',cost_o)
        np.save('./data/costdofl'+str(self.B)+'.npy',cost_f)
        
        return cost_c,cost_p,cost_o,cost_f
    
 
    
            
            
            
            
        

        
        
        