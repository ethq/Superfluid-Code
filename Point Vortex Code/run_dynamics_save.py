# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 13:14:27 2019

@author: Zak
"""

import ctypes
import PVM as pvm

# First set up initial conditions
n_vortices = 100
domain_radius = 2000
annihilate_at_radius = 1985

params = {
        'center': [1e-4, 1e-4],
        'sigma': 40
        }

cfg = pvm.Configuration(
        n_vortices,
        domain_radius,
        pvm.CONFIG_STRAT.SINGLE_CLUSTER,
        pvm.CONFIG_STRAT.CIRCS_EVEN,
        None,
        params,
        {
                'minimum_separation': 1e-1
        }
        )

T = 998
ev_config = {
    'n_vortices': n_vortices,
    'domain_radius': domain_radius,
    'annihilate_at_radius': annihilate_at_radius,
    'gamma': 0.05,
    'T': T,
    'spawn_rate': 0,
    'cfg': cfg
    }

evolver = pvm.Evolver(**ev_config)
evolver.rk()
evolver.save()

traj_data = evolver.get_trajectory_data()

analysis = pvm.Analysis(None, traj_data)

to_analyze = [
    pvm.ANALYSIS_CHOICE.CLUSTER_ANALYSIS,
    pvm.ANALYSIS_CHOICE.AUTO_CORR_CLUSTER,
    pvm.ANALYSIS_CHOICE.RMS_CLUSTER_NON_CENTERED,
    pvm.ANALYSIS_CHOICE.RMS_CLUSTER_CENTERED,
    pvm.ANALYSIS_CHOICE.RMS_NON_DIPOLE_CENTERED,
    pvm.ANALYSIS_CHOICE.RMS_NON_DIPOLE_NON_CENTERED,
    pvm.ANALYSIS_CHOICE.PAIR_CORR_W,
    pvm.ANALYSIS_CHOICE.PAIR_CORR,
    pvm.ANALYSIS_CHOICE.DIPOLE_MOMENT
    ]

analysis_data = analysis.run(to_analyze)
analysis.save()

#pc = [pvm.PlotChoice.vortices, pvm.PlotChoice.energy]
#animator = pvm.Animator(None, traj_data, analysis_data)
#animator.save_animation(pc)

# fname = f'N{n_vortices}_T{T}_S{evolver.seed}'

# plotter = pvm.HarryPlotter(fname)

# pc = [pvm.PlotChoice.rmsCluster, pvm.PlotChoice.rmsClusterNonCentered, pvm.PlotChoice.energy]

# plotter.plot(pc)

# ctypes.windll.user32.FlashWindow(ctypes.windll.kernel32.GetConsoleWindow(), True)