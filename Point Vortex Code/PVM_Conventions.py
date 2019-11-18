# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 13:21:34 2019

@author: Zak
"""


""" 
Class to enforce some conventions across files
"""

class PVM_Conventions:
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
    def save_conventions(self, max_n_vortices, T, annihilation_threshold, seed, data_type):
        atr = ("%f" % annihilation_threshold).rstrip('0')
        
        fname = "Vortex" + data_type + "_N%d_T%d_ATR" % (max_n_vortices, T) + atr + "_%d" % seed
        fname = fname + '.dat'
        
        return fname