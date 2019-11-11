# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 15:31:39 2019

@author: Zak
"""

from GPE import GPE
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation

gpe = GPE(batch_size = 200)
gpe.update()


#fig = plt.figure()
#ax = fig.gca(projection='3d')
#ax.view_init(azim=0, elev=90)
surf = plt.contourf(gpe.X, gpe.Y, np.abs(gpe.uc)**2, 100, cmap = cm.coolwarm)
plt.show()


