#Op Resp - use this code to plot op resp functions

#import packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

C = 0.4
M=25 #c=0, 0.31, 0.4

def f(r):
   #return C + (1-C)*r   #M=1 and M=2
   #return C + (1-C)*(r**2+r)/(2*(r**2-r+1))   #M=1 and M=2
   #return C + (1-C)*(3*r**2-2*r**3)   #M=3
   return C + (1-C)*(sum([binom.pmf(i,M,r) for i in range(int((M+1)/2), M+1)]))   #odd M
   #return C + (1-C)*(sum([binom.pmf(i,M,r) for i in range(int(M/2)+1,M+1)]) + 0.5*binom.pmf(int(M/2),M,r))   #even M

r = np.linspace(0, 1, 100)

#plot settings
plt.rc('axes', titlesize=20)     # fontsize of the axes title
plt.rc('axes', labelsize=21)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
plt.rc('ytick', labelsize=18)    # fontsize of the tick labels
plt.rc('legend', fontsize=20, loc='upper left')    # legend fontsize

plt.figure()
plt.plot(r, f(r), color='blue', linewidth = 3)
plt.plot(r,r, color = 'blue', alpha = 0.6)
plt.xlabel('Hearing rate, $r$')
plt.ylabel('Speaking rate, $\Psi_C(r)$')
plt.legend(['$\Psi_C(r)$', '$r$'])
#plt.title('$M\in\{1,2\}, C=0$')
#plt.title('Modified $M=2, C>0.0718$')
#plt.title('$M=3, C>0.111$')



plt.show()