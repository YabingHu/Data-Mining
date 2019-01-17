# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 16:44:08 2018

@author: yabing hu & Yang xiao
"""
#%% 
##############Problem 1
import pandas as pd 
import numpy as np  
import matplotlib.pyplot as plt
df = pd.read_csv('yahoo_ad_clicks.csv',header=None)
#df.loc[:,0] column
#df.loc[0,1] row

#%% 
############Problem 2 and 3 
############Please note that here EPX3 and MWU used same regret calculation as stochastic
############algorithms, the correct regret calculation is commented out 
def EXP3(T,k,df):
    l_hat=0
    L_hat=np.zeros(k)
    p_t=1/k*np.ones(k)
    I_t=np.zeros(T)
    count=np.zeros(k)
    exp_reward=np.zeros(k)
    temp=0
    regret=[]
    best_reward=max(df.sum(1))/T
    total_reward=0
    for t in range(T):
        eta=np.sqrt(np.log(k)/((t+1)*k))
        I_t[t]=np.random.choice(k, 1, p=p_t)
        j=int(I_t[t])
        count[j]+=1
        exp_reward[j]+=df.loc[:,t][j]
        l_hat=(1-df.loc[:,t][j])/p_t[j]
        L_hat[j]=L_hat[j]+l_hat
        p_t=np.exp(-eta*L_hat)/np.exp(-eta*L_hat).sum()
        total_reward+=df.loc[:,t][j]
        temp+=best_reward-exp_reward[j]/count[j]
        regret.append(temp/(t+1))
    print("EXP3 total reward",total_reward)
    return I_t,regret  
k,T=df.shape
I_t,regret=EXP3(T,k,df)
plt.plot(regret,label='EXP3')
plt.legend(loc='best')
plt.title('regret/t vs number of rounds')
plt.xlabel('number of rounds')
plt.ylabel('regret/t')

def THOMPSON(T,k,df):
    S=np.zeros(k)
    F=np.zeros(k)
    theta=np.zeros(k)
    I_t=np.zeros(T)
    count=np.zeros(k)
    exp_reward=np.zeros(k)
    temp=0
    regret=[]
    best_reward=max(df.sum(1))/T
    total_reward=0
    for t in range(T):
        for i in range(k):
            theta[i]=np.random.beta(S[i]+1,F[i]+1)
        arm=np.argmax(theta)
        I_t[t]=arm
        total_reward+=df.loc[:,t][arm]
        count[arm]+=1
        exp_reward[arm]+=df.loc[:,t][arm]
        r_t=df.loc[:,t][arm]
        if r_t==1:
            S[arm]+=1
        else:
            F[arm]+=1
        temp+=best_reward-exp_reward[arm]/count[arm]
        regret.append(temp/(t+1))
    print("THOMPSON total reward",total_reward)
    return I_t,regret

I_t,regret=THOMPSON(T,k,df)
plt.plot(regret,label='Thompson')
plt.legend(loc='best')
plt.title('regret/t vs number of rounds')
plt.xlabel('number of rounds')
plt.ylabel('regret/t')


def UCB1(T,k,df):
    mu=np.zeros(k)
    n=np.ones(k)
    I_t=np.zeros(T)
    UCB=np.zeros(k)
    exp_reward=np.zeros(k)
    temp=0
    regret=np.zeros(T)
    best_reward=max(df.sum(1))/T
    for i in range(k):
        I_t[i]=i
        mu[i]=df.loc[:,i][i]
        exp_reward[i]=df.loc[:,i][i]
        temp+=best_reward-exp_reward[i]
        regret[i]=temp/(i+1)
    for t in range(1,T-k+1):
        UCB=mu+np.sqrt(2*np.log(t+k)/n)
        j=np.argmax(UCB)
        I_t[k-1+t]=j
        n[j]+=1
        exp_reward[j]+=df.loc[:,t][j]
        mu[j]+=1/n[j]*(df.loc[:,t][j]-mu[j])
        temp+=best_reward-exp_reward[j]/n[j]
        regret[k-1+t]=temp/(k+t)
    print("UCB1 total reward",exp_reward.sum())
    return I_t,regret
        
I_t,regret=UCB1(T,k,df)
plt.plot(regret,label='UCB1')
plt.legend(loc='best')
plt.title('regret/t vs number of rounds')
plt.xlabel('number of rounds')
plt.ylabel('regret/t')    
    
    
def e_greedy(T,k,df):
    I_t=np.zeros(T)
    r_t=np.zeros(k)
    count=np.ones(k)
    temp=0
    regret=[]
    best_reward=max(df.sum(1))/T
    for t in range(1,T+1):
        epsilon1=1/np.log(2*t)
        epsilon2=np.random.rand(1,1)
        epsilon=min(epsilon1,epsilon2)
        if np.random.rand(1,1) < epsilon:
            I_t[t-1]= np.random.choice(k, 1)
            r_t[int(I_t[t-1])]+=df.loc[:,t-1][int(I_t[t-1])]
            count[int(I_t[t-1])]+=1
            temp+=best_reward- r_t[int(I_t[t-1])]/count[int(I_t[t-1])]
            regret.append(temp/(t+1))
        else:
            I_t[t-1]=np.argmax(r_t/count)
            r_t[int(I_t[t-1])]+=df.loc[:,t-1][int(I_t[t-1])]
            count[int(I_t[t-1])]+=1
            temp+=best_reward- r_t[int(I_t[t-1])]/count[int(I_t[t-1])]
            regret.append(temp/(t+1))
    print("e-greedy total reward",r_t.sum())
    return I_t,regret 
I_t,regret=e_greedy(T,k,df)
plt.plot(regret,label='e_greedy')
plt.legend(loc='best')
plt.title('regret/t vs number of rounds')
plt.xlabel('number of rounds')
plt.ylabel('regret/t') 

''' 
#correct way to calculate reward for non-stochastic algrithms
def EXP3(T,k,df):
    l_hat=0
    L_hat=np.zeros(k)
    loss=np.zeros(k)
    Loss=np.zeros(k)
    p_t=1/k*np.ones(k)
    I_t=np.zeros(T)
    regret=np.zeros(T)
    R_T_1=0
    total_reward=0
    for t in range(T):
        eta=np.sqrt(np.log(k)/((t+1)*k))
        I_t[t]=np.random.choice(k, 1, p=p_t)
        j=int(I_t[t])
        l_hat=(1-df.loc[:,t][j])/p_t[j]
        loss=1-df.loc[:,t]
        Loss+=loss
        L_hat[j]=L_hat[j]+l_hat
        p_t=np.exp(-eta*L_hat)/np.exp(-eta*L_hat).sum()
        R_T_1+=(loss*p_t).sum()
        regret[t]=(R_T_1-min(Loss))/(t+1)
        total_reward+=df.loc[:,t][j]
    print(total_reward)
    return I_t,regret  

k,T=df.shape
I_t,regret=EXP3(T,k,df)
plt.plot(regret,label='EXP3')
plt.legend(loc='best')
plt.title('regret/t vs number of rounds')
plt.xlabel('number of rounds')
plt.ylabel('regret/t')

      
def MWU(T,k,df):
    w=np.ones(k)
    I_t=np.zeros(T)
    Loss=np.zeros(k)
    loss=np.zeros(k)
    R_t_1=0
    regret=[]
    total_reward=0
    for t in range(T):
        p_t=w/w.sum()
        I_t[t]=(np.random.choice(k, 1, p=p_t))
        loss=1-df.loc[:,t]
        Loss+=loss
        w*=1-1/np.sqrt(T)*loss
        R_t_1+=(loss*p_t).sum()
        regret.append((R_t_1-min(Loss))/(t+1))
        total_reward+=df.loc[:,t][int(I_t[t])]
    print(total_reward)
    return I_t,regret
        
I_t2,regret2=MWU(T,k,df)
plt.plot(regret2,label='MWU')
plt.legend(loc='best')
plt.title('regret/t vs number of rounds')
plt.xlabel('number of rounds')
plt.ylabel('regret/t')
'''
  
def MWU(T,k,df):
    w=np.ones(k)
    I_t=np.zeros(T)
    Loss=np.zeros(k)
    loss=np.zeros(k)
    count=np.zeros(k)
    total_reward=0
    exp_reward=np.zeros(k)
    temp=0
    regret=[]
    best_reward=max(df.sum(1))/T
    
    for t in range(T):
        p_t=w/w.sum()
        I_t[t]=(np.random.choice(k, 1, p=p_t))
        loss=1-df.loc[:,t]
        Loss+=loss
        count[int(I_t[t])]+=1
        exp_reward[int(I_t[t])]+=df.loc[:,t][int(I_t[t])]
        w*=1-1/np.sqrt(T)*loss
        temp+=best_reward-exp_reward[int(I_t[t])]/count[int(I_t[t])]
        regret.append(temp/(t+1))
        total_reward+=df.loc[:,t][int(I_t[t])]
    print("MWU total reward",total_reward)
    return I_t,regret
        
I_t2,regret2=MWU(T,k,df)
plt.plot(regret2,label='MWU')
plt.legend(loc='best')
plt.title('regret/t vs number of rounds')
plt.xlabel('number of rounds')
plt.ylabel('regret/t')
#%% 
