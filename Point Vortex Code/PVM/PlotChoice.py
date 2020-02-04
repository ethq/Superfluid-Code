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


# IF ADDING A PLOT CHOICE, REMEMBER TO ALSO ADD TO VALIDATION LIST BELOW
class PlotChoice:
    # Only use these(in combination if desired)
    vortices = 'vortices'
    energy = 'energy'
    numberOfVortices = 'numberOfVortices'
    energyPerVortex = 'energyPerVortex'
    dipoleMoment = 'dipoleMoment'
    rmsCluster = 'rmsCluster'
    rmsNonDipoleNonCentered = 'rmsNonDipoleNonCentered'
    rmsFirstVortex = 'rmsFirstVortex'
    
    energyImageReal = 'energyImageReal'
    smallestDistance = 'smallestDistance'
    
    def show_vortex(choice):
        if type(choice) == str:
            choice = [choice]
        
        return np.array(['vortices' in c for c in choice]).any()
            
    """
    Validate a given choice
    """
    def validate_plot_choice(choice):
        # Single choice?
        if type(choice) == str:
            choice = np.array([choice])
            
        # It's important that it's a numpy array. list and numpy array have slightly 
        # different behaviour, e.g. indexing of type a[a!=b] returns the first element
        # satisfying the condition if a is a list, a numpy array returns a numpy array
        # containing _all_ elements satisfying the condition
        
        if type(choice) != np.ndarray:
            choice = np.array(choice)
        
        is_valid = np.array([c in PlotChoice.get_possible_values() for c in choice]).all()
        
        if not is_valid:
            raise ValueError('Invalid plot choice encountered.') 
    
        return choice
    
    # Replace with getattr() stuff?
    def get_possible_values():
        return [
            PlotChoice.vortices,
            PlotChoice.dipoleMoment,
            PlotChoice.energy,
            PlotChoice.numberOfVortices,
            PlotChoice.energyPerVortex,
            PlotChoice.rmsCluster,
            PlotChoice.rmsNonDipoleNonCentered,
            PlotChoice.rmsFirstVortex,
            PlotChoice.energyImageReal,
            PlotChoice.smallestDistance
            ]