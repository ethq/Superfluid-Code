# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 18:08:51 2019

@author: Z
"""


# IDs start at 0

import numpy as np
from tqdm import tqdm

class Vortex:
    iid = 0
    def __init__(self, pos, circ, t0):
        self.pos = pos
        self.circ = circ
        

        self.id = Vortex.iid
        Vortex.iid = Vortex.iid + 1
        
        self.trajectory = []
        self.trajectory.append(pos)
        
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
    
    # Returns latest position or at a time t if given
    def get_pos(self, t):
        # Adjust for time of spawning
        tid = t - self.t0
#        print(tid, self.t1)
#        assert tid >= 0 and tid < len(self.trajectory)
        
        if tid < 0 or tid >= len(self.trajectory):
            print(self.t0, t, tid)
            print(self.t1, self.id, tid, self.trajectory)
        
        return self.trajectory[tid]
    
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