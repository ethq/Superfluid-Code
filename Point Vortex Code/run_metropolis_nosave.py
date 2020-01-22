# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 11:18:42 2020

@author: Zak
"""

import PVM as pvm
    
#ev_config = {
#        'n_vortices': 30,
#        'gamma': 0.0,
#        'T': 5,
#        'spawn_rate': 0,
#        'mc_burn': 1e5,
#        'mc_steps': 1e5
#        }

#evolver = pvm.Evolver(**ev_config)
#evolver.metropolis()
#
#traj_data = evolver.get_trajectory_data()
#analysis = pvm.Analysis(None, traj_data)
#
#analysis_data = analysis.full_analysis()
#
#animator = pvm.Animator(None, traj_data, analysis_data)
#animator.show_animation(pvm.PlotChoice.vortices_energy)

cfg = {
       'n_vortices': 100,
       'temperature': 1e15,
       'bbox_ratio': 50,
       'vorticity_tol': 1e-3,
       'annihilation_threshold': 1e-2,
       'domain_radius': 1,
       'skip': 1,
       'total_steps': 1e7
       }

e = pvm.Evolver_MCMC(**cfg)
e.evolve()