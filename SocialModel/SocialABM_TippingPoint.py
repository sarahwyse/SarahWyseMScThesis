#social ABM - plot tipping point for various C - in parallel with binary search

#import packages
import numpy as np
import random as rnd
import matplotlib.pyplot as plt
from collections import Counter
import time
from joblib import Parallel, delayed

t1 = time.time()

#define parameters
N = 500      #total population
C_min = 0      #committed minority proportion
C_max = 0.5
M = np.arange(1,5,1)         #memory length
t_end = 100   #number of timesteps to run simulation for
tipping = np.zeros((len(M),2)) #track tipping point for each memory length


error = 10/N  #find tipping point to within 10 agents


def social_binary_search(mem, C_min, C_max, C):
    C_mid = (C_min + C_max)/2
    # If C_temp is too far from C, not within error bound, continue
    while abs(C-C_mid) > error:
        #set initial condition, 0 represents no climate mitigation, 1 represents climate action
        # start with uncommitted with memory=0 and committed with memory=1
        c_pop = int(C_mid*N)
        Opinion = np.zeros(N-c_pop) #used to track opinions of uncommitted population over time
        Opinion_sum = np.zeros(N)   #used to make computing new opinion more efficient
        tempMem = 0
        Memory = np.zeros((N,mem))
        for i in range(c_pop):
            Opinion_sum[i] = mem
            for j in range(mem):
                Memory[i,j] = 1
    
        for t in range(t_end): #run model for t_end*(N/2) interactions
            for inter in range(int(N/2)):
                i, j = rnd.sample(range(0,N),2) #choose two agents, i speaks, j listens
                if j>=c_pop: #make sure listener is not committed
                    #determine new memory based on speaker opinion
                    if Opinion_sum[i]==mem/2: #in case speaker is undecided, sample uniformly from both opinions
                        tempMem = rnd.randint(0,1)
                    elif Opinion_sum[i]<mem/2:
                        tempMem = 0
                    else:
                        tempMem = 1
                    #compute new Opinion_avg for listener and add listener back into matrix
                    Opinion_sum[j] = Opinion_sum[j] - Memory[j,0] + tempMem
                    Memory[j,:] = np.append(Memory[j,1:mem], tempMem)
                    #convert from memory to opinion for listener, have to use j-C since Opinion only tracks uncommitted agents
                    if Opinion_sum[j]==mem/2: #in case listener is undecided, sample uniformly from both opinions
                        Opinion[j-c_pop] = rnd.randint(0,1)
                    elif Opinion_sum[j]<mem/2: 
                        Opinion[j-c_pop] = 0
                    else:
                        Opinion[j-c_pop] = 1
        #If convention overturned, tipping point at smaller value of C
        C = C_mid
        if Counter(Opinion)[1]/(N-C_mid)>0.5:
            C_max = C_mid    
        # Else convention persists, tipping point at larger value of C
        else:
            C_min = C_mid 
        C_mid = (C_min + C_max)/2
    #return memory length and committed minority proportion once within error bound
    return mem, C    
        
                        
tipping = Parallel(n_jobs=14)(delayed(social_binary_search)(mem, C_min, C_max, 1) for mem in M)

    
#plot the results
plt.figure()
plt.plot(*zip(*tipping),'.',color='blue')
plt.xlabel('Memory length, M')
plt.ylabel('Committed minority, C')
#plt.title(r'N=%i, ' %N + r' and t=%i' %t_end)

plt.rc('axes', titlesize=20)     # fontsize of the axes title
plt.rc('axes', labelsize=15)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
plt.rc('legend', fontsize=13)    # legend fontsize

plt.ylim(0.0, 0.5)

plt.show()

t2 = time.time() - t1

