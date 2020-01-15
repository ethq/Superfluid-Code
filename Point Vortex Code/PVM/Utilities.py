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
        return cart2pol_scalar(x,y)

# Given two points o1(x,y), o2(x,y) and a radius R, determines:
#   - the point of collision with the disk
#   - the reflection of o2-o1 about the unit normal of the disk
# It is assumed that the centre of the disk is the origin
        
# If get_cp is set, the tuple(reflected vector, collision point) is returned instead of just the vector
def reflect(o1, o2, R, get_cp = False):
    o1 = np.array(o1)
    o2 = np.array(o2)
    
    # We first find the point of collision    
    do = np.linalg.norm(o2-o1)
    
    # Scale factor for the connecting vector o2-o1
    alpha = 1/do**2*(-(np.dot(o1, o2) - np.dot(o1, o1)) + np.sqrt(do**2*R**2 + np.dot(o1, o2)**2 - np.dot(o1, o1)*np.dot(o2, o2)))
    
    # Point of collision
    p = o1 + alpha*(o2-o1)
    
    # Length of reflected vector
    lref = do - np.linalg.norm(p-o2)
    
    # Unit normal at collision point
    n = -p/np.linalg.norm(p)
    
    # Vector to be reflected
    v1 = p - o1
    
    # Reflected vector
    v2 = v1 - 2*n*(1-np.dot(v1,n))
    
    # Fix the length
    v2 = v2/np.linalg.norm(v2)*lref
    
    # And done
    if get_cp:
        return (v2, p)
    return v2
    

def cart2pol_scalar(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)
    
# Takes a hexadecimal string and spits out the corresp RGB tuple, normalized to [0, 1]
def hex2one(h):
    h = h.lstrip('#')
    rgb = tuple(int(h[i:i+2], 16)/255 for i in (0, 2, 4))
    return rgb
    
# Shitty alternative to numpy.linalg.norm - also, this returns the square... which I have used several places ugh
# TODO: clean up all occurences of this crap in project
def eucl_dist(v1, v2):
    d = v1-v2
    return d @ d.T

# Currently vortex indices match up to indices in the vortex array, as they are added sequentially
# May change in future, hence function
def get_vortex_by_id(vortices, id_):
    if type(id_) == int:
        id_ = [id_]
        
    return [v for v in vortices if v.id in id_]

"""
Vortices: [array] of class Vortex
tid: [integer] time to get position
with_id: if supplied, also returns an array of vortex ids


Returns:
    (positions, circulations, [ids])

"""
def get_active_vortex_cfg(vortices, tid):
    # Mask alive vortices
    mask = [v.is_alive(tid) for v in vortices]
    
    a_vortices = vortices[mask]
    
    # Get ids
    ids = np.array([v.id for v in a_vortices])
    
    # Get positions
    pos = np.array([v.get_pos(tid) for v in a_vortices])
    
    # Get circulations
    circ = np.array([v.circ for v in a_vortices])
    
    return {
            'positions': pos,
            'circulations': circ,
            'ids': ids
            }

# Returns active vortices at time tid. By default finds all vortices that are currently active
def get_active_vortices(vortices, tid = -np.Infinity):
    # Create mask
    mask = [v.is_alive(tid) for v in vortices]
    
    return vortices[mask]
    