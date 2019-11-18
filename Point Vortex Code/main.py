# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 13:14:27 2019

@author: Zak
"""

from PVM_Evolver import PVM_Evolver
from PVM_Analysis import PVM_Analysis
from PVM_Animation import PVM_Animation

evolver = PVM_Evolver()
evolver.rk4()

traj_data = evolver.get_trajectory_data()
analysis = PVM_Analysis(None, traj_data)

analysis_data = analysis.cluster_analysis()

animator = PVM_Animation(None, traj_data, analysis_data)
animator.animate_trajectories()