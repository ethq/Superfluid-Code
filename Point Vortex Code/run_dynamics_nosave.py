# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 13:14:27 2019

@author: Zak
"""

import PVM as pvm
    
ev_config = {
        'n_vortices': 10,
        'gamma': 0.0,
        'T': 2,
        'spawn_rate': 0,
        'coords': pvm.INIT_STRATEGY.DOUBLE_CLUSTER
        }

evolver = pvm.Evolver(**ev_config)
evolver.rk4()

traj_data = evolver.get_trajectory_data()
analysis = pvm.Analysis(None, traj_data)

analysis_data = analysis.full_analysis()

animator = pvm.Animator(None, traj_data, analysis_data)
animator.show_animation(pvm.PlotChoice.vortices_energy)