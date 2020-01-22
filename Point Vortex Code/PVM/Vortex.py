# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 18:08:51 2019

@author: Z
"""


# IDs start at 0

import numpy as np
from tqdm import tqdm
from PVM.Utilities import eucl_dist


"""

Simple class to keep track of vortex positions, ids & trajectories

pos:     [[x,y]] [Float] Initial position
circ:    [Integer] Circulation
t0:      [Integer] Frame of spawning
max_traj_len: [Integer] How long trajectories can be. For memory problems.

"""

def image_pos(pos, R):
    # Square magnitude
    r2 = eucl_dist(pos, np.array([0,0]))
   
    # .. and image position
    impos = R**2*pos/r2
    
    return impos

class Vortex:
    iid = 0
    def __init__(self, pos, circ, t0, max_traj_len = 10000):
        self.pos = pos
        self.circ = circ
        

        self.id = Vortex.iid
        Vortex.iid = Vortex.iid + 1
        
        self.trajectory = []
        self.trajectory.append(pos)
        
        self.max_traj_len = max_traj_len
        
        # Time of spawning
        self.t0 = t0
        
        # Time of annihilation
        self.t1 = np.Inf
        
    def set_pos(self, pos):
        # Vortex annihilated
        if self.t1 != np.Inf:
            return
        
        self.pos = pos
        self.trajectory.append(pos)
        
    def pop_pos(self):
        if len(self.trajectory) > 1:
            self.pos = self.trajectory[-2]
        else:
            self.pos = [0, 0]
        return self.trajectory.pop()
    
    
    """
    Returns the position of this vortex's image at time t. 
    Assumes for the moment that the vortex lives in a circular domain of radius R
    """
    def get_impos(self, t, R):        
        return image_pos(self.get_pos(t), R)
    
    # Returns latest position or at a time t if given
    def get_pos(self, t = -1):
        if t < 0:
            return self.trajectory[-1]
        # Adjust for time of spawning
        tid = t - self.t0
        assert tid >= 0 and tid < len(self.trajectory)
        
        return np.array(self.trajectory[tid])
    
    # Returns whether this vortex is active. If t1 > tc, then it has not yet
    # been annihilated
    def is_alive(self, tc):
        if tc >= self.t0 and tc < self.t1:
            return True
        
        return False
    
    def annihilate(self, t1):
        tqdm.write('Oh no I died at t = %d and my name is %d' % (t1, self.id))
        self.t1 = t1
    
    """
        
    Returns part of trajectory between times [tend-tlen, tend]
    
    """
    def get_trajectory(self, tend = -1, tlen = 5):
        # Return from end of trajecory
        if tend == -1:
            return np.array(self.trajectory[-tlen:])
        
        # Adjust for spawn time
        tend = tend - self.t0
        
        # Find start time
        ts = np.max([tend-tlen,0])
        
        return np.array(self.trajectory[ts:tend])
    
        

if __name__ == '__main__':
    v1 = Vortex([0,0], 1)
    v2 = Vortex([0,0], 1)
    print(v1.id, v2.id, Vortex([0,0], -1).id)