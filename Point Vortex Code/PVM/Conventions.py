# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 13:21:34 2019

@author: Zak
"""


""" 
Class to enforce some conventions across files
"""

from .Utilities import hex2one

class Conventions:
    def __init__(self):
        pass
    
    """
    Colour scheme.
    

    WARNING!!!! Animator.py uses setattr while looping over the scheme to set class variables.
    
    Hence it may be a litle dangerous to modify this fellow...

    """
    def colour_scheme():
        return {
            'vortex_colours': {-1: (*hex2one('#bd2b2b'), 0.7), 1: (*hex2one('#383535'), 0.7)},
            'dipole_colour': '#c0e39d',
            'cluster_colour': '#57769c'
            }
    
    
    """
    Consistent way of naming data files from evolution/analysis using metadata
    
    max_vortices: [integer] metadata
    T:            [integer] metadata
    annihilation_threshold: [float] metadata
    seed:         [integer] metadata
    
    data_type: [string] can be either 'Evolution' or 'Analysis'
    
    """
    def save_conventions(max_n_vortices, T, annihilation_threshold, seed, data_type, plot_choice = None, conv = 'short'):
        atr = ("%f" % annihilation_threshold).rstrip('0')
        
        # Convention 1: lots of info in filename
        if conv == 'long':
            fname = "_N%d_T%d_ATR" % (max_n_vortices, T) + atr + "_%d" % seed
        # Convention 2: less info in filename
        elif conv == 'short':
            fname = "_N%d_T%d_S%d" % (max_n_vortices, T, seed)
        elif conv == 'seed':
            fname = "_S%d" % seed
        else:
            raise ValueError('Unknown convention in PVM.Conventions')
        
        # Select appropriate folder and file extension
        if data_type == 'Evolution' or data_type == 'Analysis':
            fname = "Datafiles/" + data_type + fname + '.dat'
        elif data_type == 'Animation':
            if len(plot_choice):
                fname = '-'.join(plot_choice) + fname
            fname = "Animations/" + fname + '.mp4'
        else:
            raise ValueError('unknown data_type in Conventions.save_conventions()')
        
        return fname
        