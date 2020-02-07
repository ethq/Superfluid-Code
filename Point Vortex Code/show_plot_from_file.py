# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 10:42:32 2019

@author: Zak
"""

import PVM as pvm

# fname = 'N20_T50_S768390681'
# fname = 'N20_T50_S457173602'
# fname = 'N20_T50_S869893185'
# fname = 'N26_T50_S717109192'
# fname = 'N10_T50_S87655771'
# fname = 'N10_T50_S996866482' ### Chiral
# fname = 'N10_T50_S873349814' ### Chiral
# fname = 'N10_T150_S393963592'  ### Chiral

# fname = 'N20_T500_S402701135' ### Mixed
# fname = 'N30_T500_S144692810'  ### Mixed. Shows that (pure) clusters at least do not follow t/sqrt(t) scaling.
# fname = 'N40_T500_S517932362' ### Mixed


# fname = 'N50_T500_S836575032' ### Mixed
# fname = 'N50_T500_S308947746' ### Mixed
# fname = 'N50_T500_S752728417' ### Mixed
fname = 'N50_T500_S853514746' ### Mixed
fname = 'N50_T500_S189428672' ### Mixed


# fname = 'N30_T500_S550439413' ### Chiral


plotter = pvm.HarryPlotter(fname)

pc = [pvm.PlotChoice.rmsCluster, pvm.PlotChoice.rmsNonDipoleNonCentered]
# pc = [pvm.PlotChoice.rmsCluster]

plotter.plot(pc)

# plotter.plot_cfg(percent = 80)