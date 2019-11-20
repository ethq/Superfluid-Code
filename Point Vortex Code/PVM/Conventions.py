# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 13:21:34 2019

@author: Zak
"""


""" 
Class to enforce some conventions across files
"""

class Conventions:
    def __init__(self):
        pass
    
    """
    
    Consistent way of naming data files from evolution/analysis using metadata
    
    max_vortices: [integer] metadata
    T:            [integer] metadata
    annihilation_threshold: [float] metadata
    seed:         [integer] metadata
    
    data_type: [string] can be either 'Evolution' or 'Analysis'
    
    """
    def save_conventions(max_n_vortices, T, annihilation_threshold, seed, data_type, conv = 'short'):
        atr = ("%f" % annihilation_threshold).rstrip('0')
        
        # Convention 1: lots of info in filename
        if conv == 'long':
            fname = data_type + "_N%d_T%d_ATR" % (max_n_vortices, T) + atr + "_%d" % seed
        # Convention 2: less info in filename
        elif conv == 'short':
            fname = data_type + "_N%d_T%d_S%d" % (max_n_vortices, T, seed)
        elif conv == 'seed':
            fname = data_type + "_S%d" % seed
        
        else:
            raise ValueError('Unknown convention in PVM.Conventions')
        
        # Select appropriate folder and file extension
        if data_type == 'Evolution' or data_type == 'Analysis':
            fname = "Datafiles/" + fname + '.dat'
        elif data_type == 'Animation':
            fname = "Animations/" + fname + '.mp4'
        else:
            raise ValueError('unknown data_type in Conventions.save_conventions()')
        
        return fname
        