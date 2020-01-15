# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 15:11:55 2019

@author: Zak
"""

import matplotlib.pyplot as plt
from matplotlib import animation
from numpy import random 
from PVM.Utilities import reflect, cart2pol, pol2cart
import numpy as np

x0 = [.2, .2]
r = cart2pol(np.array([x0]))


x1 = np.array([1.2, .2])
r1 = cart2pol(np.array([x1]))


(r2, p) = reflect(x0, x1, 1, True)
r2 = cart2pol(np.array([r2]))

rp = cart2pol(np.array([p]))
print(np.linalg.norm(rp))

plt.polar(r[0][1], r[0][0], 'o')
plt.polar(rp[0][1], rp[0][0], 'o')
plt.polar([r[0][1], r1[0][1]], [r[0][0], r1[0][0]], '-')
plt.polar([rp[0][1], r2[0][1]], [rp[0][0], r2[0][0]], '-')
plt.polar(r1[0][1], r1[0][0], 'o')
plt.polar(r2[0][1], r2[0][0], 'o')
plt.show()



#fig = plt.figure()
#ax1 = plt.axes(xlim=(-108, -104), ylim=(31,34))
##line, = ax1.plot([], [], lw=2)
#plt.xlabel('Longitude')
#plt.ylabel('Latitude')
#
#plotlays, plotcols = [2], ["black","red"]
#lines = []
#for index in range(2):
#    lobj = ax1.plot([],[],lw=2,color=plotcols[index])[0]
#    lines.append(lobj)
#
#
#def init():
#    for line in lines:
#        line.set_data([],[])
#    return lines
#
#x1,y1 = [],[]
#x2,y2 = [],[]
#
## fake data
#frame_num = 100
#gps_data = [-104 - (4 * random.rand(2, frame_num)), 31 + (3 * random.rand(2, frame_num))]
#
#
#def animate(i):
#
#    x = gps_data[0][0, i]
#    y = gps_data[1][0, i]
#    x1.append(x)
#    y1.append(y)
#
#    x = gps_data[0][1,i]
#    y = gps_data[1][1,i]
#    x2.append(x)
#    y2.append(y)
#
#    xlist = [x1, x2]
#    ylist = [y1, y2]
#
#    #for index in range(0,1):
#    for lnum,line in enumerate(lines):
#        line.set_data(xlist[lnum], ylist[lnum]) # set data for each line separately. 
#
#    return lines
#
## call the animator.  blit=True means only re-draw the parts that have changed.
#anim = animation.FuncAnimation(fig, animate, init_func=init,
#                               frames=frame_num, interval=10, blit=True)
#
#
#plt.show()