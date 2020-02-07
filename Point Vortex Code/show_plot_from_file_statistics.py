# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 10:19:02 2020

@author: Zak
"""

import PVM as pvm
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.stats as st


lc = (*pvm.Utilities.hex2one('#bd2b2b'), 0.3)
lb = (*pvm.Utilities.hex2one('#383535'), 1)

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


# Load fnames - use save_metadata.py to get a set of seeds satisfying certain criteria
seedf = 'Metadata/N50_T500_Mixed.dat'
with open(seedf, 'rb') as f:
    seeds = pickle.load(f)
    
fnames = ['N50_T500_S' + str(s) for s in seeds]

# What are we looking at?
statistic = 'rmsCluster'
# statistic = 'rmsNonDipole'
# statistic = 'rmsClusterNonCentered'
# statistic = 'rmsNonDipoleNonCentered'
# statistic = 'auto_corr'

# statistics = ['rmsCluster', 'rmsClusterNonCentered']

vals = []
t = 1 + np.arange(50000) #### Assumes all seeds have fixed T = 500, dt = .1

for f in tqdm(fnames):
    fname_analysis = 'Datafiles/Analysis_' + f + '.dat'    
    # fname_evolution = 'Datafiles/Evolution_' + f + '.dat'
    
    # ef = open(fname_evolution, "rb")
    af = open(fname_analysis, "rb")
            
    # evolution_data = pickle.load(ef)
    analysis_data = pickle.load(af)

    # settings = evolution_data['settings']
    
    # Some files may not have it(can extend their analysis)
    if not statistic in analysis_data:# or not statistics[1] in analysis_data:
        tqdm.write(f'{f} did not contain {statistic}')
        continue
    
    # Add statistic - the sum here is over all vortices, leaving us with the statistic as f(t)
    # vals.append( 
    #     np.array([[np.sum(d) for d in analysis_data[statistics[1]]]]).flatten()/t  
    #     - np.array([[np.sum(d) for d in analysis_data[statistics[0]]]]).flatten()/t
    #     )
    
    vals.append( np.array([[np.mean(d) for d in analysis_data[statistic]]]).flatten() )
    
    # For autocorrelation, sum has already been done. Could perhaps wait, so we have single-particle autocorrelations?
    # vals.append( analysis_data[statistic] - analysis_data[statistic][0])
    
    # Plot it
    plt.plot(t, vals[-1], color = lc)
    
    
    af.close()
    # ef.close()
    
# If centered, add some small values at t = 0 to give non-infinite confidence intervals
# for v in vals:
    # v[0] = np.random.normal(0, 1e-6)
    
    
# Calculate its 95% confidence interval
cfid = st.t.interval(0.95, len(vals)-1, loc=np.mean(vals, axis = 0), scale=st.sem(vals, axis = 0))

# Plot it
plt.fill_between(t, cfid[0], cfid[1], zorder = 1e3)


# Get the average
avg = np.mean(vals, axis = 0).flatten()

plt.plot(t, avg, label = 'Average', color = lb)
# plt.legend()

plt.xlabel('Time')

plt.show()





# plotter = pvm.HarryPlotter(fname)

# pc = [pvm.PlotChoice.rmsCluster, pvm.PlotChoice.rmsNonDipoleNonCentered, pvm.PlotChoice.energy]
# pc = [pvm.PlotChoice.rmsCluster]

# plotter.plot(pc)

# plotter.plot_cfg(percent = 50)