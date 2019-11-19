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
    vortices_energy = 'Vortices_Energy'
    vortices_numberOfVortices = 'Vortices_NumberOfVortices'
    vortices_energyPerVortex = 'Vortices_EnergyPerVortex'
    
    
    def get_possible_values():
        return [
            PlotChoice.vortices_energy,
            PlotChoice.vortices_energyPerVortex,
            PlotChoice.vortices_numberOfVortices
            ]
        
    def choice2axis_setup(c):
        # Map to axis setup functions in Animator class
        # Kept here to avoid having it as a class property
        choice2axis = {
            'Vortices_Energy': ['axsetup_vortices', 'axsetup_energy'],
            'Vortices_NumberOfVortices': ['axsetup_vortices', 'axsetup_energyPerVortex'],
            'Vortices_EnergyPerVortex': ['axsetup_vortices', 'axsetup_numberOfVortices']
            }
        
        # Shouldn't be possible to get this far with an incorrect choice but we'll check anyway
        if not c in choice2axis.keys():
            raise ValueError('Invalid choice encountered in PlotChoice.choice2axis_setup()')
            
        return choice2axis[c]