# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 10:01:15 2020

@author: Zak
"""

import numpy as np

t1 = np.array([-1, 5])
t2 = np.array([4, -3])
t3 = np.array([-20, 2])
t4 = np.array([5, 6])


ts = np.array([t1, t2, t3, t4])

autocorr = np.sum([t*t for t in ts])

rms = np.sum([np.linalg.norm(t)**2 for t in ts])

ac = np.sum([np.dot(t,t) for t in ts])