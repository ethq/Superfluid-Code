# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 10:02:00 2019

@author: Zak
"""

import PVM as pvm
import ctypes

# Assumes the seeds have been evolved and analyzed

fnames = ['N30_T100_S95642',
          'N668_T50_S68869',
          'N150_T50_S67603',
          'N30_T50_S92586'
          ]

for fname in fnames:
    animator = pvm.Animator(fname)
    animator.save_animation(pvm.PlotChoice.vortices_energy)
    