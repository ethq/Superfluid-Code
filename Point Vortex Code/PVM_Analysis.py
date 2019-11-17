# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 14:28:04 2019

@author: Z
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.animation as animation
from itertools import chain, compress
from tqdm import tqdm
import collections
import pickle

from Utilities import pol2cart, cart2pol, eucl_dist, get_active_vortices
from Vortex import Vortex


"""
    Expects a filename, file of format given by PVM_Evolver:
        data = {
            'Settings': settings,
            'Trajectories': self.trajectories,
            'Circulations': self.circulations,
            'Vortices': self.vortices
            }
        
        where 
        
        settings = {
                'Total_time': self.T,
                'Timestep': self.dt,
                'N_steps': int(self.T/self.dt),
                'Domain_radius': self.domain_radius,
                'Annihilation_threshold': self.annihilation_threshold,
                'Tolerance': self.tol
                } 
        
        Trajectories is an (N_steps, N_vortices, 2) array.
        Vortices is an array consisting of Vortex classes - dead or alive
        Circulations is an (N_steps, N_vortices, N_vortices) array. 
"""
class PVM_Analysis:
    
    def __init__(self, fname):
        
        
        with open(fname, "rb") as f:
            data = pickle.load(f)
            self.settings = data['Settings']
            self.vortices = data['Vortices']
            self.trajectories = ['Trajectories']
            self.circulations = ['Circulations']
            
        
        self.dipoles = []
        self.clusters = []
        
        self.cluster_analysis()
        
    
    # Currently vortex indices match up to indices in the vortex array, as they are added sequentially
    # May change in future, hence function
    def get_vortex_by_id(self, id_):
        return self.vortices[id]
    
    # Run a complete cluster analysis for all time frames
    # find_dipoles must be ran prior to find_clusters; two parts of the algorithm
    def cluster_analysis(self):
        for i in np.arange(self.settings['N_steps']):
            
            self.dipoles.append( self.find_dipoles(i) )
            self.clusters.append( self.find_clusters(i) )
            
    
    def find_clusters(self, i):
        vortices = get_active_vortices(self.vortices, i)
        
        
        clusters = []
        cluster_ids = []

        for v in vortices:
            if v.id in dipoles or v.id in cluster_ids:
                continue

            # Find cluster partners
            tar = self.get_vortex_by_id(v.id)
            
            #
            _, tar_enemy_dist = self.find_nn(vortices, v.id, i, True)

            cluster = []

            for v2 in vortices:
                # Skip if dipole or of opposite sign. Assumption that dipoles have been previously computed for this timestep
                if v2.id in self.dipoles[-1] or v2.circ != tar.circ or v.id == v2.id:
                    continue

                # We found a same-sign not-dipole vortex.
                # Check their distance and their distance to nearest opposite sign

                dist_friend = self.eucl_dist(tar, cfg[0][j, :])

                if (tar_enemy_dist < dist_friend):
                    continue

                _, dist_enemy = self.find_nn(cfg, j, True)

                if (dist_enemy < dist_friend):
                    continue

                # Friends are closer than either are to an enemy, cluster them
                if not len(cluster):
                    cluster.append(i)

                cluster.append(j)

            cluster_ids.extend(cluster)
            if len(cluster):
                clusters.append(cluster)

        return clusters

    """
        Rule: if two vorticies are mutually nearest neighbours, we classify them as a dipole
    """
    def find_dipoles(self, i):
        vortices = get_active_vortices(self.vortices, i)
        
        dipoles = []
        
        # Loop over all vortices
        for v in vortices:
            # If already classified as dipole, skip
            if v.id in dipoles:
                continue
            
            # Find nearest neighbour of this vortex
            nn_id, _ = self.find_nn(vortices, v.id, i)
            
            # And nearest neighbour of nearest neighbour
            nn_nn_id, _ = self.find_nn(vortices, nn_id, i)
            
            # Mutual nearest neighbour found, classify as dipole if signs match
            if ((nn_nn_id == v.id) and v.circ == self.get_vortex_by_id(nn_nn_id).circ):
                dipoles.append(v.id)
                dipoles.append(nn_nn_id)
                
        # Todo maybe add dipoles in ... pairs? That would be logical! 
        return dipoles

    """
    Find nearest neighbour of vortex number vid at time i. if opposite is true, find only neighbour of opposite circulation
    """
    
    def find_nn(self, vortices, vid, i, opposite = False):
        smallest_dist = np.Inf
        
        # Target to find nearest neighbour of
        tar = self.get_vortex_by_id(vid)
        
        # Nearest neighbour id
        nn = -1
        
        # Loop over all vortices
        for v in vortices:
            # If this vortex is the one we are looking at, skip
            # If opposite is true and these are same sign, skip
            if vid == v.id or (opposite and v.circ == tar.circ):
                continue
            
            dist = eucl_dist(tar.get_pos(i), v.get_pos(i))
            
            if (dist < smallest_dist):
                smallest_dist = dist
                nn = v.id
                
        return nn, smallest_dist


    
    
if __name__ == '__main__':
    pvm = PVM_Analysis('VortexEvolution_N5_T5_ATR0.01.dat')
#    fname = 'VortexEvolution_N5_T5_ATR0.01.dat'
#    with open(fname, "rb") as f:
#        data = pickle.load(f)