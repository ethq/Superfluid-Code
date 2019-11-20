# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 14:25:14 2019

@author: Zak
"""

"""

Convenience class to keep track of what the Animator class can plot/animate. 
Helps dynamically determine which functions Animator uses to setup the relevant axes/layout

"""
class PlotChoice:
    vortices_energy = 'vortices_energy'
    vortices_numberOfVortices = 'vortices_numberOfVortices'
    vortices_energyPerVortex = 'vortices_energyPerVortex'
    
    
    def get_possible_values():
        return [
            PlotChoice.vortices_energy,
            PlotChoice.vortices_energyPerVortex,
            PlotChoice.vortices_numberOfVortices
            ]