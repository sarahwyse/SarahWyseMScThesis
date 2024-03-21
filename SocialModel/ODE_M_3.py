#social ODE model with M=3

import numpy as np
import scipy.integrate as int
import matplotlib.pyplot as plt
#import seaborn as sns  # to make plots pretty

#model
def M3(t, z):
    AAA, AAB, ABA, BAA, ABB, BAB, BBA, BBB = z
    return [AAB*(AAA+AAB+ABA+BAA+C) - AAA*(BBB+BBA+BAB+ABB),   #AAA
            ABB*(AAA+AAB+ABA+BAA+C) + ABA*(AAA+AAB+ABA+BAA+C) - AAB*(AAA+AAB+ABA+BAA+C) - AAB*(BBB+BBA+BAB+ABB),  #AAB
            BAB*(AAA+AAB+ABA+BAA+C) + BAA*(AAA+AAB+ABA+BAA+C) - ABA*(AAA+AAB+ABA+BAA+C) - ABA*(BBB+BBA+BAB+ABB),  #ABA
            AAA*(BBB+BBA+BAB+ABB) + AAB*(BBB+BBA+BAB+ABB) - BAA*(AAA+AAB+ABA+BAA+C) - BAA*(BBB+BBA+BAB+ABB),  #BAA
            BBA*(AAA+AAB+ABA+BAA+C) + BBB*(AAA+AAB+ABA+BAA+C) - ABB*(AAA+AAB+ABA+BAA+C) - ABB*(BBB+BBA+BAB+ABB),  #ABB
            ABB*(BBB+BBA+BAB+ABB) + ABA*(BBB+BBA+BAB+ABB) - BAB*(AAA+AAB+ABA+BAA+C) - BAB*(BBB+BBA+BAB+ABB),  #BAB
            BAA*(BBB+BBA+BAB+ABB) + BAB*(BBB+BBA+BAB+ABB) - BBA*(AAA+AAB+ABA+BAA+C) - BBA*(BBB+BBA+BAB+ABB),  #BBA
            BBA*(BBB+BBA+BAB+ABB) - BBB*(AAA+AAB+ABA+BAA+C)]   #BBB

#run simulation
C = 0.11
y0 = [0, 0, 0, 0, 0, 0, 0, 1-C]     #replace 1 with 1-C to include committed minority in total population
sol = int.solve_ivp(M3, [0, 150], y0, method='LSODA')



#plot
cmap = plt.get_cmap('brg')
colors = cmap(np.linspace(0,0.5,8))
fig, ax = plt.subplots()
ax.set_prop_cycle(color=colors)

plt.plot(sol.t, sol.y.T, linewidth = 3)   
plt.xlabel('Time')
plt.ylabel('Uncommitted proportions')
plt.legend([r'$X_{AAA}$', '$X_{AAB}$', '$X_{ABA}$', '$X_{BAA}$', '$X_{ABB}$', '$X_{BAB}$', '$X_{BBA}$', '$X_{BBB}$'])
plt.title(r'$M=3$ and $C=0.11$')
          
plt.rc('axes', titlesize=20)     # fontsize of the axes title
plt.rc('axes', labelsize=23)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
plt.rc('ytick', labelsize=18)    # fontsize of the tick labels
plt.rc('legend', fontsize=20, loc='upper right')    # legend fontsize

plt.ylim(-0.05, 1.05)

plt.show()
