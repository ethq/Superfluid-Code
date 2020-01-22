# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 10:02:00 2019

@author: Zak
"""

import PVM as pvm
import ctypes

# Assumes the seeds have been evolved and analyzed

fnames = ['N20_T20_S30819'
          ]

for fname in fnames:
    animator = pvm.Animator(fname)
    animator.save_animation(pvm.PlotChoice.vortices_energy)
    