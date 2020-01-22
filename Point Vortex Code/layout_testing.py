# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 15:56:04 2020

@author: Zak
"""

import matplotlib.pyplot as plt

f = plt.figure()

f.add_subplot(3,1,1)
f.add_subplot(3,1,2)
f.add_subplot(3,1,3)

plt.tight_layout()

plt.show()


fig = plt.figure()
fig.add_subplot(1, 2, 1)   #top and bottom left
fig.add_subplot(2, 2, 2)   #top right
fig.add_subplot(2, 2, 4)   #bottom right 
plt.show()

f = plt.figure()

f.add_subplot(1, 2, 1)
f.add_subplot(3, 2, 2)
f.add_subplot(3, 2, 4)
f.add_subplot(3, 2, 6)

plt.tight_layout()
plt.show()