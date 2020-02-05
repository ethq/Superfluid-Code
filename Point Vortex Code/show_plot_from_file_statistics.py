# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 10:19:02 2020

@author: Zak
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 10:42:32 2019

@author: Zak
"""

import PVM as pvm
import pickle
import numpy as np
import matplotlib.pyplot as plt

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


fnames = ['N50_T500_S836575032', ### Mixed
          'N50_T500_S308947746', ### Mixed
          'N50_T500_S752728417', ### Mixed
          'N40_T500_S517932362',
          # 'N30_T500_S144692810',
          'N50_T500_S560206210',
          'N50_T500_S853514746'
          # 'N20_T500_S402701135'
          ]

nonDipoleNonCenteredRms2 = []
t = 1 + np.arange(50000)

for f in fnames:
    fname_analysis = 'Datafiles/Analysis_' + f + '.dat'    
    fname_evolution = 'Datafiles/Evolution_' + f + '.dat'
    
    ef = open(fname_evolution, "rb")
    af = open(fname_analysis, "rb")
            
    evolution_data = pickle.load(ef)
    analysis_data = pickle.load(af)

    settings = evolution_data['settings']
    
    nonDipoleNonCenteredRms2.append( [[np.sum(d) for d in analysis_data['rmsNonDipoleNonCentered']]] )
    
    plt.plot(t, np.array(nonDipoleNonCenteredRms2[-1]).flatten()/t, label = f"R: {settings['domain_radius']}, N: {settings['max_n_vortices']}")
    
    
    af.close()
    ef.close()
    
ndncrms2 = np.mean(nonDipoleNonCenteredRms2, axis = 0).flatten()



plt.plot(t, ndncrms2/t, label = 'Average')
plt.legend()
plt.show()
# plotter = pvm.HarryPlotter(fname)

# pc = [pvm.PlotChoice.rmsCluster, pvm.PlotChoice.rmsNonDipoleNonCentered, pvm.PlotChoice.energy]
# pc = [pvm.PlotChoice.rmsCluster]

# plotter.plot(pc)

# plotter.plot_cfg(percent = 50)