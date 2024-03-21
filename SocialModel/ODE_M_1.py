#social ODE model with M=1

import numpy as np
import scipy.integrate as int
import matplotlib.pyplot as plt

#model
def M1(t, z):    
    A, B = z
    return [B*(A+C) - A*B,
            A*B - B*(A+C)]

#run simulation
C = 0.05
y0 = [0, 1-C]     #replace 1 with 1-C to include committed minority in total population
sol = int.solve_ivp(M1, [0, 150], y0, method='LSODA')

#plot
cmap = plt.get_cmap('brg')
colors = cmap(np.linspace(0,0.5,2))
fig, ax = plt.subplots()
ax.set_prop_cycle(color=colors)

plt.plot(sol.t, sol.y.T, linewidth = 3)
plt.xlabel('Time')
plt.ylabel('Uncommitted proportions')  #Proportion of opinions
plt.legend([r'$X_A$', '$X_B$'])
plt.title(r'$M=1$ and $C=0$')

plt.rc('axes', titlesize=20)     # fontsize of the axes title
plt.rc('axes', labelsize=23)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
plt.rc('ytick', labelsize=18)    # fontsize of the tick labels
plt.rc('legend', fontsize=20, loc='upper right')    # legend fontsize

plt.ylim(-0.05, 1.05)

plt.show()
