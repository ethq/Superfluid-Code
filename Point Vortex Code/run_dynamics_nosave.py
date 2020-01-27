# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 13:14:27 2019

@author: Zak
"""

import ctypes
import PVM as pvm

# First set up initial conditions
n_vortices = 6
domain_radius = 1

params = {
        'center': [1e-4, 1e-4],
        'sigma': .4
        }

cfg = pvm.Configuration(
        n_vortices,
        domain_radius,
        pvm.CONFIG_STRAT.SINGLE_CLUSTER,
        pvm.CONFIG_STRAT.CIRCS_EVEN,
        18817436,
        params
        )

ev_config = {
    'n_vortices': n_vortices,
    'domain_radius': domain_radius,
    'gamma': 0,
    'T': 10,
    'spawn_rate': 0,
    'cfg': cfg
    }

evolver = pvm.Evolver(**ev_config)
evolver.rk4()

traj_data = evolver.get_trajectory_data()

analysis = pvm.Analysis(None, traj_data)
analysis_data = analysis.full_analysis()

pc = [pvm.PlotChoice.vortices, pvm.PlotChoice.energy]
animator = pvm.Animator(None, traj_data, analysis_data)
animator.show_animation(pc)

ctypes.windll.user32.FlashWindow(ctypes.windll.kernel32.GetConsoleWindow(), True)