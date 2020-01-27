# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 12:36:50 2019

@author: Zak
"""

import PVM as pvm

#fname = "N30_T50_S92586"
fname = 'N10_T10_S61145'
animator = pvm.Animator(fname)
animator.show_animation([pvm.PlotChoice.vortices, pvm.PlotChoice.rmsCluster])