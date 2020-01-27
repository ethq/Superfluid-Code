# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 10:42:32 2019

@author: Zak
"""

import PVM as pvm

fname = 'N20_T10_S18817436'
plotter = pvm.HarryPlotter(fname)

pc = [pvm.PlotChoice.rmsFirstVortex, pvm.PlotChoice.numberOfVortices]

plotter.plot(pc)