# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 15:29:51 2019

@author: Zak
"""

import numpy as np

"""
     Class mainly exists to store parameters for time-dependent potential,
     Also to easily add different forms of V(t) in the future without 
     changing the main code.
     
     type_:  [String] Type of stirring potential. Supports 'none', 'gaussian'
     
     params_: [Dict] For type_ = 'gaussian', expects:
         strength: number denoting strength of the potential
         width: number giving the width of the gaussian
         radius: number, it is the radius of the stirrers orbit. 
         velocity: the velocity with which the stirrer moves
         
    grid: [Tuple] First component taken to be X, second taken to be Y
"""
class Stirrer():
    def __init__(self, type_, params_, grid):
        if (type_ == 'gaussian'):
            self.vs0 = params_['strength']
            self.w = params_['width']
            
            self.rad = params_['radius']
            
            self.velocity = params_['velocity']
            self.omega = self.velocity/self.rad
            
            self.x = grid[0]
            self.y = grid[1]
            
            self.V = self.v_gauss
            
        if (type == 'none'):
            self.V = self.v_none
    
    def v_none(self, t):
        return 0
    
    # Return time-dependent potential
    def v_gauss(self, t):
        return self.vs0*np.exp(-1/(self.w**2)*(self.x - self.rad*np.cos(self.omega*t))**2 - 1/(self.w**2)*(self.y + self.rad*np.sin(self.omega*t))**2)