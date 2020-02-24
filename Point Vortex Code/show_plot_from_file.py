# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 10:42:32 2019

@author: Zak
"""

import PVM as pvm
import os
import numpy as np

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
fname = 'N100_T500_S915560108'
fname = 'N70_T5000_S920411951'

# fname = 'N70_T300_S87240089'
# fname = 'N70_T300_S422500560'
# fname = 'N70_T300_S662417938'
# fname = 'N70_T2000_S874026731'
# fname = 'N70_T5000_S41244916'
fname = 'N100_T508_S955505076'
fname = 'N100_T508_S943535723'

# fname = 'N100_T5082_S573711462'
fname = 'N100_T504_S786990155'
fname = 'N100_T512_S704409062'
fname = 'N100_T512_S909890148'


fname = 'N50_T515_S605700655'

fname = 'N50_T519_S683367893'
fname = 'N50_T519_S690640270'

fname = 'N50_T520_S711558627'

# fname = 'N30_T500_S550439413' ### Chiral

fname = 'N50_T520_S828427048'
fname = 'N50_T5_S828427048'
fname = 'N50_T5000_S828427048'

fname = 'N2_T666_S412311452'
fname = 'N2_T666_S261883477'
fname = 'N2_T66_S889078464'
fname = 'N2_T266_S759875717'
fname = 'N2_T1066_S264611168'
fname = 'N50_T5066_S554509817'
fname = 'N50_T5014_S496387584'
fname = 'N50_T5014_S496387584'
fname = 'N50_T5014_S819913311'

fname = 'N50_T5015_R2000_G0.1_S532745994'

# fname = 'N50_T15000_R2000_G0_S858518372'
fname = 'N50_T15000_R2000_G0_S907960698'
fname = 'N50_T35000_R2000_G0_S554620699'

plotter = pvm.HarryPlotter(fname)

pc = [pvm.PlotChoice.rmsCluster, 
       # pvm.PlotChoice.energy
       # pvm.PlotChoice.rmsClusterNonCentered,
      # pvm.PlotChoice.rmsNonDipole, 
      # pvm.PlotChoice.rmsNonDipoleNonCentered,
      # pvm.PlotChoice.numberOfVortices,
       # pvm.PlotChoice.auto_corr,
       # pvm.PlotChoice.auto_corr_cluster,
       # pvm.PlotChoice.auto_corr_nondipole
      ]

# plotter.plot(pc)
slideshow = True


# Save a slideshow
if slideshow:
    for i in np.linspace(0, 99, 100):
        plotter.plot_cfg(percent = i, save = True)