#social ODE model with M=2

import numpy as np
import scipy.integrate as int
import matplotlib.pyplot as plt
#import seaborn as sns  # to make plots pretty

#model
def M2(t, z):
    AA, AB, BA, BB = z
    return [-AA*(BB+0.5*AB+0.5*BA) + AB*(AA+0.5*AB+0.5*BA+C),
            -AB*(AA+0.5*AB+0.5*BA+C) + BB*(AA+0.5*AB+0.5*BA+C) - AB*(BB+0.5*AB+0.5*BA) + BA*(AA+0.5*AB+0.5*BA+C),
            -BA*(BB+0.5*AB+0.5*BA) + AA*(BB+0.5*AB+0.5*BA) - BA*(AA+0.5*AB+0.5*BA+C) + AB*(BB+0.5*AB+0.5*BA),
            -BB*(AA+0.5*AB+0.5*BA+C) + BA*(BB+0.5*AB+0.5*BA)]

#run simulation
C = 0.2
y0 = [0, 0, 0, 1-C]     #replace 1 with 1-C to include committed minority in total population
sol = int.solve_ivp(M2, [0, 150], y0, method='LSODA')



#plot
cmap = plt.get_cmap('brg')
colors = cmap(np.linspace(0,0.5,4))
fig, ax = plt.subplots()
ax.set_prop_cycle(color=colors)

plt.plot(sol.t, sol.y.T, linewidth = 3)
plt.xlabel('Time')
plt.ylabel('Uncommitted proportions')
plt.legend([r'$X_{AA}$', '$X_{AB}$', '$X_{BA}$', '$X_{BB}$'])
plt.title(r'$M=2$ and $C=0$')

plt.rc('axes', titlesize=20)     # fontsize of the axes title
plt.rc('axes', labelsize=23)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
plt.rc('ytick', labelsize=18)    # fontsize of the tick labels
plt.rc('legend', fontsize=20, loc='upper right')    # legend fontsize

plt.ylim(-0.05, 1.05)

plt.show()
