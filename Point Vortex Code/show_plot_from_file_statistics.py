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
import subprocess

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score


lc = (*pvm.Utilities.hex2one('#bd2b2b'), 0.1)
lb = (*pvm.Utilities.hex2one('#383535'), 1)
lg = (*pvm.Utilities.hex2one('#bde364'), 1)
lo = (*pvm.Utilities.hex2one('#e88317'), 1)

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

N = 50
T = 35000#5019
R = 2000
G = 0

subprocess.call(f"python save_metadata.py {T} {N} {R} {G}")

# Load fnames - use save_metadata.py to get a set of seeds satisfying certain criteria
seedf = f"Metadata/N{N}_T{T}_Mixed.dat"
with open(seedf, 'rb') as f:
    seeds = pickle.load(f)
    
fnames = [f"N{N}_T{T}_R{R}_G{G}_S" + str(s) for s in seeds]
print(seeds)

# What are we looking at?
# statistic = 'rmsCluster'
statistic = 'rmsNonDipole'
statistic = 'rmsClusterNonCentered'
statistic = 'rmsNonDipoleNonCentered'
# statistic = 'auto_corr'

# statistics = ['rmsCluster', 'rmsClusterNonCentered']

vals = []

for f in tqdm(fnames):
    fname_analysis = 'Datafiles/A_' + f + '.dat'    
    # fname_evolution = 'Datafiles/Evolution_' + f + '.dat'
    
    # ef = open(fname_evolution, "rb")
    af = open(fname_analysis, "rb")
            
    # evolution_data = pickle.load(ef)
    analysis_data = pickle.load(af)
    
    nl = len(analysis_data['dipoles'])
    dt = T/nl
    t = (1 + np.arange(nl))*dt #### Assumes all seeds have fixed T = 500, dt = .1

    # settings = evolution_data['settings']
    
    # Some files may not have it(can extend their analysis)
    if not statistic in analysis_data:# or not statistics[1] in analysis_data:
        tqdm.write(f'{f} did not contain {statistic}')
        continue
    
    # Mask out values > 1950 - a little below the annihilation threshold
    # We do this because the annihilate-at-boundary code strands one dipole partner in no mans land
    # (typically relevant only for non-dipole clustering)
    stat = analysis_data[statistic]
    stat2 = []
    for d in stat:
        r = np.sqrt(np.array(d))
        r = r < 1900
        stat2.append(np.array(d)[r])
            
    stat = stat2
    
    # Add statistic - the sum here is over all vortices, leaving us with the statistic as f(t)
    # vals.append( 
    #     np.array([[np.sum(d) for d in analysis_data[statistics[1]]]]).flatten()/t  
    #     - np.array([[np.sum(d) for d in analysis_data[statistics[0]]]]).flatten()/t
    #     )
    #- np.mean(analysis_data[statistic][0]) ## abs, subtraction are "hacks"
    vals.append( np.array([np.sqrt(np.abs(np.mean(d) - np.mean(stat[0])))  for d in stat]).flatten() )
    
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
plt.fill_between(t, cfid[0], cfid[1], zorder = 1e3, alpha = .3)


# Get the average
avg = np.mean(vals, axis = 0).flatten()

# # Let's also make some fits to the average
modela = make_pipeline(PolynomialFeatures(1), LinearRegression())
modelb = make_pipeline(PolynomialFeatures(1), LinearRegression())

# Introduce a cut; we do expect a linear trend only after the initial transient
cutb = 10000
tb = t[:cutb]
modelb.fit(t[:cutb, np.newaxis], avg[:cutb, np.newaxis]**2)

avg_predb = modelb.predict(t[:, np.newaxis])
# if cutb:
    # avg_predb = np.concatenate([avg_predb, np.mean(avg[cutb:])*np.ones(len(t)-cutb)[:, np.newaxis]])

    
# avg_predb = np.concatenate( [np.mean(avg[:cutb])*np.ones(cutb)[:, np.newaxis],  modelb.predict(t[cutb:, np.newaxis])] )

mseb = mean_squared_error(avg[:cutb]**2, avg_predb[:cutb])
r2b = r2_score(avg[:cutb]**2, avg_predb[:cutb])

# Fit a quadratic
cuta = 10000
ta = t[cuta:]
modela.fit(t[cuta:, np.newaxis], avg[cuta:, np.newaxis])

avg_preda = modela.predict(t[cuta:, np.newaxis]) 
# if cuta:
    # avg_preda = np.concatenate([np.mean(avg[:cuta])*np.ones(cuta)[:, np.newaxis], avg_preda])

msea = mean_squared_error(avg[cuta:], avg_preda)
r2a = r2_score(avg[cuta:], avg_preda)

print(f"Square root fit: MSE = {mseb}, R2 = {r2b}\nLinear fit: MSE = {msea}, R2 = {r2a}")

# print(f"Quadratic parameters: {modela[1].coef_[0][1]}t + {modela[1].coef_[0][2]}t^2")

# plt.plot(t, modela[1].intercept_ + modela[1].coef_[0][1]*t + modela[1].coef_[0][2]*t**2, label = 'manual pred')
plt.plot(t, avg, label = 'Average', color = lb, zorder = 1e2)
# plt.plot(t, avg/t, label = 'Inva', color = lo, zorder = 1e5)
plt.plot(ta, avg_preda, label = 'Linear fit', color = lg, zorder = 5e3)
plt.plot(t, np.sqrt(avg_predb), label = 'Square root fit', color = lo, zorder = 5e3)
plt.legend()

plt.xlabel('Time')
plt.title(statistic)

plt.show()





# plotter = pvm.HarryPlotter(fname)

# pc = [pvm.PlotChoice.rmsCluster, pvm.PlotChoice.rmsNonDipoleNonCentered, pvm.PlotChoice.energy]
# pc = [pvm.PlotChoice.rmsCluster]

# plotter.plot(pc)

# plotter.plot_cfg(percent = 50)