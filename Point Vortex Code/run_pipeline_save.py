# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 13:14:27 2019

@author: Zak
"""

from PVM.Evolver import Evolver
from PVM.Analysis import Analysis
from PVM.Animator import Animator

evolver = Evolver()
evolver.rk4()
evolver.save()

traj_data = evolver.get_trajectory_data()


analysis = Analysis(None, traj_data)
analysis_data = analysis.full_analysis()
analysis.save()

animator = Animator(None, traj_data, analysis_data)
animator.animate_trajectories()