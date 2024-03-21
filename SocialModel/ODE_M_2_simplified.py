#social ODE model with M=2 simplified

import numpy as np
import scipy.integrate as int
import matplotlib.pyplot as plt

#model
def M2_simp(t, z):
    AA, U, BB = z
    return [-AA*(BB+0.5*U) + U*(AA+0.5*U+C),
            AA*(BB+0.5*U) - U*(AA+0.5*U+C) + BB*(AA+0.5*U+C) - U*(BB+0.5*U),
            -BB*(AA+0.5*U+C) + U*(BB+0.5*U)]

#run simulation
C = 0.4
y0 = [0, 0, 1-C]     #replace 1 with 1-C to include committed minority in total population
sol = int.solve_ivp(M2_simp, [0, 150], y0, method='LSODA')



#plot
cmap = plt.get_cmap('brg')
colors = cmap(np.linspace(0,0.5,3))
fig, ax = plt.subplots()
ax.set_prop_cycle(color=colors)

plt.plot(sol.t, sol.y.T, linewidth = 3)
plt.xlabel('Time')
plt.ylabel('Uncommitted proportions')
plt.legend([r'$X_{AA}$', '$X_{U}$', '$X_{BB}$'])
plt.title(r'Modified $M=2$ and $C=0.07$')

plt.rc('axes', titlesize=20)     # fontsize of the axes title
plt.rc('axes', labelsize=23)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
plt.rc('ytick', labelsize=18)    # fontsize of the tick labels
plt.rc('legend', fontsize=20, loc='upper right')    # legend fontsize

plt.ylim(-0.05, 1.05)

plt.show()