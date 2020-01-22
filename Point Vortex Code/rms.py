# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 14:08:31 2020

@author: Zak
"""

import ctypes
import PVM as pvm

ev_config = {
    'n_vortices': 10,
    'domain_radius': 100,
    'gamma': 0.00,
    'T': 10,
    'spawn_rate': 0,
    'coords': pvm.INIT_STRATEGY.SINGLE_CLUSTER
    }

evolver = pvm.Evolver(**ev_config)
evolver.rk4()
evolver.save()

traj_data = evolver.get_trajectory_data()

analysis = pvm.Analysis(None, traj_data)
analysis_data = analysis.full_analysis()
analysis.save()

#animator = pvm.Animator(None, traj_data, analysis_data)
#animator.save_animation(pvm.PlotChoice.vortices_energy)

ctypes.windll.user32.FlashWindow(ctypes.windll.kernel32.GetConsoleWindow(), True)