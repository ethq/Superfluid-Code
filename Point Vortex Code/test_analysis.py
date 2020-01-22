# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 13:50:28 2020

@author: Zak
"""

import PVM as pvm
import numpy as np
import matplotlib.pyplot as plt

from paircorrelation import pairCorrelationFunction_2D


from PVM.Utilities import pol2cart, cart2pol, eucl_dist, get_active_vortices, get_active_vortex_cfg, plot_cfg
from PVM.Vortex import Vortex

## Test weighted_pair_correlation() from Analysis class
def test_wpc():
    
    # We first run a simple simulation w/ uniformly distributed vortices
    # Only really to pass params to Analysis
    ev_config = {
        'n_vortices': 50000,
        'domain_radius': 20,
        'gamma': 0.0,
        'T': .01,
        'spawn_rate': 0,
        'coords': pvm.INIT_STRATEGY.UNIFORM,
        'circ': pvm.INIT_STRATEGY.CIRCS_EVEN
        }

    evolver = pvm.Evolver(**ev_config)
    evolver.rk4()
    
    traj_data = evolver.get_trajectory_data()
    analysis = pvm.Analysis(None, traj_data)
    
    # Initial config is supposed to be uniform
    cfg = get_active_vortex_cfg(analysis.vortices, 0)
    
#    plot_cfg(cfg)
    
    # Grab pair corr, number of vortices, density
#    g, n, rho, bins = analysis.get_weighted_pair_corr(cfg['positions'], 0)
    g, r = analysis.get_pair_corr3(cfg, weighted = True)
#    print(g,bins)
    # Normalize
#    g = g/(rho)
    
#    g,r, _ = pairCorrelationFunction_2D(cfg['positions'][:,0], cfg['positions'][:, 1], 20, 5, 0.1)
    
    # And let's see if it is indeed close to one as it should be
#    print(g)
    
    plt.plot(r[:], g[:], 'o')
    plt.show()
    
    return g, r
    
if __name__ == '__main__':
    g,bins = test_wpc()
    print(np.mean(g))
    