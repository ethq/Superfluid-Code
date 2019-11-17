# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 18:08:51 2019

@author: Z
"""


# IDs start at 0

import numpy as np

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
        self.t1 = -1
        
    def set_pos(self, pos):
        # Vortex annihilated
        if self.t1 != -1:
            return
        
        self.pos = pos
        self.trajectory.append(pos)
    
    # Returns latest position or at a time t if given
    def get_pos(self, t = -1):
        if t == -1:
            return self.pos
        
        # Adjust for time of spawning
        tid = t - self.t0
        assert tid > 0 and tid < len(self.trajectory)
        
        return self.trajectory[tid]
    
    # Returns whether this vortex is active. If t1 > tc, then it has not yet
    # been annihilated
    def is_alive(self, tc = np.Inf):
        # No annihilation has ever occured; must be alive
        if self.t1 == -1:
            return True
        
        # Annihilation occured, but if it in the future of tc, it is alive at t = tc
        return self.t1 > tc
    
    def annihilate(self, t1):
        print('Oh no I died')
        self.t1 = t1
    
    """
        
    Returns part of trajectory between times [tend-tlen, tend]
    
    """
    def get_trajectory(self, tend = -1, tlen = 5):
        # Return from end of trajecory
        if tend == -1:
            return self.trajectory[-tlen:]
        
        # Find start time
        ts = np.max([tend-tlen,0])
        
        return self.trajectory[ts:ts+tlen]
    
        

if __name__ == '__main__':
    v1 = Vortex([0,0], 1)
    v2 = Vortex([0,0], 1)
    print(v1.id, v2.id, Vortex([0,0], -1).id)