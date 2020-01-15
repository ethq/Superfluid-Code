# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 13:14:27 2019

@author: Zak
"""

import ctypes
import PVM as pvm

ev_config = {
    'n_vortices': 150,
    'gamma': 0.02,
    'T': 80,
    'spawn_rate': 0
    }

evolver = pvm.Evolver(**ev_config)
evolver.rk4()
evolver.save()

traj_data = evolver.get_trajectory_data()

analysis = pvm.Analysis(None, traj_data)
analysis_data = analysis.full_analysis()
analysis.save()

animator = pvm.Animator(None, traj_data, analysis_data)
animator.save_animation(pvm.PlotChoice.vortices_energy)

ctypes.windll.user32.FlashWindow(ctypes.windll.kernel32.GetConsoleWindow(), True)