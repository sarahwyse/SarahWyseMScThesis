# social opinion response functions - tipping point

import numpy as np
import math
from scipy.stats import binom
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import time

t1 = time.time()

M = np.arange(3,4,1)         #memory length
error = 1e-12  #error bound in root finding and tipping point search
r_L = 0     #initial leftmost root
r_R = 0.5   #initial middle root (right root of the ones we solve for)

tipping = np.zeros((len(M),2)) #track tipping point for each memory length

C_min = 0      #committed minority proportion bounds for binary search
C_max = 0.5


def social_binary_search(M, C_min, C_max, C):
    C_mid = (C_min + C_max)/2
    # If C_temp is too far from C, not within error bound, continue
    while abs(C-C_mid) > error:
        #get initial phi value for r0
        if (M%2):
            phi0L = C_mid + (1-C_mid)*(sum([binom.pmf(i,M,r_L) for i in range(int((M+1)/2), M+1)])) - r_L
            phi0R = C_mid + (1-C_mid)*(sum([binom.pmf(i,M,r_R) for i in range(int((M+1)/2), M+1)])) - r_R
        else:
            phi0L = C_mid + (1-C_mid)*(sum([binom.pmf(i,M,r_L) for i in range(int(M/2)+1, M+1)]) + 0.5*binom.pmf(int(M/2),M,r_L)) - r_L
            phi0R = C_mid + (1-C_mid)*(sum([binom.pmf(i,M,r_R) for i in range(int(M/2)+1, M+1)]) + 0.5*binom.pmf(int(M/2),M,r_R)) - r_R
            
        #get roots
        root0 = root_finder(M,C_mid,r_L,2*error,phi0L)
        root1 = root_finder(M,C_mid,r_R,r_R-2*error,phi0R) 
        #print(root0,root1)
        # If roots are nan or bigger than 0.5, tipping point at smaller value of C (>0.5 necessary for M=1 since it finds the only A root otherwise)
        C = C_mid
        if math.isnan(root0) or math.isnan(root1) or (root0>0.5) or (root1>0.5):
            C_max = C_mid    
        # Else real roots found, tipping point at larger value of C
        else:            
            C_min = C_mid
        C_mid = (C_min + C_max)/2    
    #return memory length and committed minority proportion once within error bound
    return M, C    
        
def root_finder(M,C,r0,r1,phi0):
    #repeat root_finder until values of r are close enough to each other
    while (abs(r0-r1)>=error):
        #compute phi(r1)
        if (M%2):   #M%2 gets remainder of M/2, if 1 then M is odd and this statement is true        
            phi1 = C + (1-C)*(sum([binom.pmf(i,M,r1) for i in range(int((M+1)/2), M+1)])) - r1
            #need +1 since python does not include last value in range
        else:
            phi1 = C + (1-C)*(sum([binom.pmf(i,M,r1) for i in range(int(M/2)+1,M+1)]) + 0.5*binom.pmf(int(M/2),M,r1)) - r1
        #compute new r value for next loop
        new_r = r1 - (phi1*(r1-r0))/(phi1-phi0)    #use secant method to get new r value
        #get new parameter values for next loop
        phi0 = phi1
        r0 = r1
        r1 = new_r
    #return root once error condition is met
    return r1

#tipping = social_binary_search(1,C_min,C_max,1)                     
tipping = Parallel(n_jobs=14)(delayed(social_binary_search)(mem, C_min, C_max, 1) for mem in M)

    
#plot the results
plt.figure()
plt.plot(*zip(*tipping),'.',color='blue')
plt.xlabel('Memory length, M')
plt.ylabel('Committed minority, C')

plt.rc('axes', titlesize=20)     # fontsize of the axes title
plt.rc('axes', labelsize=16)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=13)    # fontsize of the tick labels
plt.rc('ytick', labelsize=13)    # fontsize of the tick labels
plt.rc('legend', fontsize=14)    # legend fontsize

plt.ylim(0, 0.5)

plt.show()

t2 = time.time() - t1