# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 10:42:32 2019

@author: Zak
"""

import PVM as pvm

# seed: 851125867 has an energy gain of ~10 using rk4. Energy gain is the same with rk5. circulations all positive
# seed: 919776311 is balanced.
# seed: 179172593 is balanced, and image/real energy differentials are mirrored 
# seed: 768390681 is pure positive, deviation ~ 68    shows extremely odd behaviour of image energies
# seed: 128234509 is pure positive, deviation ~ 1.7

# Whats the difference between these two seeds? 


# fname = 'N20_T50_S768390681'
# fname = 'N20_T50_S457173602'
# fname = 'N20_T50_S869893185'
# fname = 'N26_T50_S717109192'
# fname = 'N10_T50_S87655771'
# fname = 'N10_T50_S996866482'
# fname = 'N10_T50_S873349814'
# fname = 'N10_T150_S393963592'
fname = 'N20_T500_S402701135'
# fname = 'N30_T500_S144692810'  ### Mixed. Shows that (pure) clusters at least do not follow t/sqrt(t) scaling.
# fname = 'N40_T500_S517932362'
# fname = 'N30_T500_S550439413'


plotter = pvm.HarryPlotter(fname)

# pc = [pvm.PlotChoice.rmsCluster, pvm.PlotChoice.rmsNonDipoleNonCentered, pvm.PlotChoice.energy]
# pc = [pvm.PlotChoice.rmsCluster]

# plotter.plot(pc)

plotter.plot_cfg(percent = 50)