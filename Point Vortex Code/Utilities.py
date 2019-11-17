# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 14:51:44 2019

@author: Z
"""

import numpy as np

"""
Converts cartesian to polar coordinates. Returns [r, theta].
If y is not supplied, array MUST BE of form
x = [
        [x0, y0],
        [x1, y1],
        ...
    ]
"""
def cart2pol(x, y = None):
    # If y is not supplied, we provide some other behaviour
    if np.any(y == None):
        out = []
        for p in x:
            out.append( list(cart2pol_scalar(p[0], p[1])) )
        return out
    else:
        return self.cart2pol_scalar(x,y)

def cart2pol_scalar(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)
    
def eucl_dist(v1, v2):
    d = v1-v2
    return d @ d.T


# Returns active vortices at time tid. By default finds all vortices that are currently active
def get_active_vortices(vortices, tid = -np.Infinity):
    # Create mask
    mask = [v.is_alive(tid) for v in vortices]
    return vortices[mask]