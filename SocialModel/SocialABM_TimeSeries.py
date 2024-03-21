#social ABM - fixed memory

#import packages
import numpy as np
import random as rnd
import matplotlib.pyplot as plt
from collections import Counter
import time

t1 = time.time()

#define parameters
N = 1000      #total population
C_prop = 0.11   #committed minority proportion
C = int(C_prop*N)        #committed minority
M = 3         #memory length
t_end = 1000   #number of timesteps to run simulation for
interactions = int(t_end*N/2) #number of interactions to run model for, each agent has ~1 interaction per timestep
prop = np.zeros((2,interactions+1)) #track proportion of opinions over time
prop[:,0] = [1-C_prop,0]

Opinion = np.zeros(N-C)
Opinion_avg = np.zeros(N)
tempMem = 0

#set initial condition, 0 represents no climate mitigation, 1 represents climate action
# start with uncommitted with memory=0 and committed with memory=1
Memory = np.zeros((N,M))
for i in range(C):
    Opinion_avg[i] = 1
    for t in range(M):
        Memory[i,t] = 1

#run the model
for t in range(t_end): #run model for t_end*(N/2) interactions
    for n in range(int(N/2)):
        i, j = rnd.sample(range(0,N),2) #choose two agents, i speaks, j listens
        if j>=C: #make sure listener is not committed
            #determine new memory based on speaker opinion
            if Opinion_avg[i]==0.5: #in case speaker is undecided, sample uniformly from both opinions
                tempMem = rnd.randint(0,1)
            elif Opinion_avg[i]<0.5:
                tempMem = 0
            else:
                tempMem = 1
            #compute new Opinion_avg for listener and add listener back into matrix
            Opinion_avg[j] = Opinion_avg[j] - (Memory[j,0]/M) + (tempMem/M)
            Memory[j,:] = np.append(Memory[j,1:M], tempMem)
            #convert from memory to opinion for listener, have to use j-C since Opinion only tracks uncommitted agents
            if Opinion_avg[j]==0.5: #in case listener is undecided, sample uniformly from both opinions
                Opinion[j-C] = rnd.randint(0,1)
            elif Opinion_avg[j]<0.5: 
                Opinion[j-C] = 0
            else:
                Opinion[j-C] = 1
    #count proportion of opinion
    count = Counter(Opinion)
    prop[:,t+1] = [count[0]/N,count[1]/N] #of form [count(0), count(1)]
    
#plot the results
plt.figure()
plt.plot(1,prop[1,1],'-',color='blue') #count ones (A)
plt.plot(1,prop[0,1],'-',color='red') #count zeros (B)
for t in range(t_end+1):
    plt.plot(t,prop[1,t],'.',color='blue') #count ones (A)
    plt.plot(t,prop[0,t],'.',color='red') #count zeros (B)
plt.xlabel('Time')
plt.ylabel('Proportion of opinions')
plt.legend(['A', 'B'])
plt.title(r'M=%i' %M + r' and C=%1.2f' %C_prop)

plt.rc('axes', titlesize=20)     # fontsize of the axes title
plt.rc('axes', labelsize=23)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
plt.rc('ytick', labelsize=18)    # fontsize of the tick labels
plt.rc('legend', fontsize=20, loc='upper right')    # legend fontsize

plt.ylim(-0.05, 1.05)

plt.show()

t2 = time.time() - t1