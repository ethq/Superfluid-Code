# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 10:02:00 2019

@author: Zak
"""

import PVM as pvm

fname = 'N30_T100_S95642'
animator = pvm.Animator(fname, animate_trails = False)
animator.save_animation(pvm.PlotChoice.vortices_energy)