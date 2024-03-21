#social Opinion response functions - bifurcation diagrams for various M on one plot

import numpy as np
import math
from scipy.stats import binom
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

mem = [15, 30, 45, 70] #change legend below if changing memory lengths plotted
cmap = plt.get_cmap('Blues')
mem_colour = cmap(np.linspace(0.4,1,len(mem)+2)) #extra colours for M=2 and M->infty

dC = 0.0001
C = np.arange(dC,1,dC) #start C slightly bigger than 0
error = 1/1000
r_L = 0     #initial leftmost root
r_R = 0.5   #initial middle root (right root of the ones we solve for)

plt.figure()
#included to get correct legend
for i in range(len(mem)+2):
    plt.plot(0.5,-1, '.',color = mem_colour[i])
plt.plot(0.5,-1, '.',color = "black")

def bifur(j, M):   
    for c in C:
        #get initial values of phi for both left and right root so that we calculate only one phi value each function call
        if (M[j]%2):
            phi0L = c + (1-c)*(sum([binom.pmf(i,M[j],r_L) for i in range(int((M[j]+1)/2), M[j]+1)])) - r_L
            phi0R = c + (1-c)*(sum([binom.pmf(i,M[j],r_R) for i in range(int((M[j]+1)/2), M[j]+1)])) - r_R
        else:
            phi0L = c + (1-c)*(sum([binom.pmf(i,M[j],r_L) for i in range(int(M[j]/2)+1, M[j]+1)]) + 0.5*binom.pmf(int(M[j]/2),M[j],r_L)) - r_L
            phi0R = c + (1-c)*(sum([binom.pmf(i,M[j],r_R) for i in range(int(M[j]/2)+1, M[j]+1)]) + 0.5*binom.pmf(int(M[j]/2),M[j],r_R)) - r_R
            
        #get leftmost root and plot it
        root0 = root_finder(M[j],c,r_L,2*error,phi0L)
        plt.plot(c,root0,'.',color = mem_colour[j+1])
        #use same function to find middle root and plot it
        root1 = root_finder(M[j],c,r_R,r_R-2*error,phi0R)
        plt.plot(c,root1,'.',color = mem_colour[j+1])  #middle will always be unstable
        
        #break loop once roots are nan
        if math.isnan(root0) or math.isnan(root1):
            break
        
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

for j in range(len(mem)):
    bifur(j,mem) 

#plot M=2 with combined undecided individuals
for c in C:
    roots = fsolve(lambda r: c + (1-c)*(r**2+r)/(2*(r**2-r+1)) - r, [0,1/2])
    plt.plot(c,roots[0],'.',color = mem_colour[0])
    plt.plot(c,roots[1],'.',color = mem_colour[0])
    if abs(roots[0]-roots[1])<error:
        break

#plot M->infty
plt.plot(C,np.repeat(1,len(C)), '.', color = "black")
plt.plot(C[0:int(len(C)/2)],np.repeat(0.5,int(len(C)/2)),'.',color=mem_colour[-1])
plt.plot(C[0:int(len(C)/2)],C[0:int(len(C)/2)],'.',color=mem_colour[-1])

#plot settings
plt.rc('axes', titlesize=20)     # fontsize of the axes title
plt.rc('axes', labelsize=16)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=13)    # fontsize of the tick labels
plt.rc('ytick', labelsize=13)    # fontsize of the tick labels
plt.rc('legend', fontsize=14)    # legend fontsize

plt.xlabel('Committed minority, $\mathcal{C}_\mathcal{M}$')
plt.ylabel('Proportion holding opinion A')
#plt.legend(['M=3', 'M=15', 'M=30', 'M=70', "M $\\to \infty$", 'A only']) 
plt.legend(['M=2*', 'M=15', 'M=30', 'M=45', 'M=70', "M $\\to \infty$", 'A only'])  



plt.ylim(0, 1.05)

plt.show()