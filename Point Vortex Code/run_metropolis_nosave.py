# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 11:18:42 2020

@author: Zak
"""

import PVM as pvm
    
ev_config = {
        'n_vortices': 30,
        'gamma': 0.0,
        'T': 5,
        'spawn_rate': 0
        }

evolver = pvm.Evolver(**ev_config)
evolver.rk4()

traj_data = evolver.get_trajectory_data()
analysis = pvm.Analysis(None, traj_data)

analysis_data = analysis.full_analysis()

animator = pvm.Animator(None, traj_data, analysis_data)
animator.show_animation(pvm.PlotChoice.vortices_energy)