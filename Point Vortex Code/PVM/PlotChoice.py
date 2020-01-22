# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 14:25:14 2019

@author: Zak
"""

"""

Convenience class to keep track of what the Animator class can plot/animate. 
Helps dynamically determine which functions Animator uses to setup the relevant axes/layout

"""

"""

TODO: Much better to hardcode to integers, then use a map if a str rep is needed

"""

import numpy as np


class PlotChoice:
    vortices = 'vortices'
    vortices_energy = 'vortices_energy'
    vortices_numberOfVortices = 'vortices_numberOfVortices'
    vortices_energyPerVortex = 'vortices_energyPerVortex'
    vortices_dipoleMoment = 'vortices_dipoleMoment'
    vortices_rmsCluster = 'vortices_rmsCluster'
    
    energy = 'energy'
    numberOfVortices = 'numberOfVortices'
    energyPerVortex = 'energyPerVortex'
    dipoleMoment = 'dipoleMoment'
    rmsCluster = 'rmsCluster'
    
    def show_vortex(choice):
        if type(choice) == str:
            choice = [choice]
        
        return np.array(['vortices' in c for c in choice]).any()
            
        
    
    def get_possible_values():
        return [
            PlotChoice.vortices_energy,
            PlotChoice.vortices_energyPerVortex,
            PlotChoice.vortices_numberOfVortices
            ]