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

from Utilities import pol2cart, cart2pol, eucl_dist, get_active_vortices, get_active_vortex_cfg
from Vortex import Vortex

from PVM_Conventions import PVM_Conventions


"""
    Expects a filename, file of format given by PVM_Evolver:
        data = {
            'settings': settings,
            'trajectories': self.trajectories,
            'circulations': self.circulations,
            'vortices': self.vortices
            }
        
        where 
        
        settings = {
                'total_time': self.T,
                'timestep': self.dt,
                'n_steps': int(self.T/self.dt),
                'domain_radius': self.domain_radius,
                'annihilation_threshold': self.annihilation_threshold,
                'tolerance': self.tol
                } 
        
        Trajectories is an (N_steps, N_vortices, 2) array.
        Vortices is an array consisting of Vortex classes - dead or alive
        Circulations is an (N_steps, N_vortices, N_vortices) array. 
"""
class PVM_Analysis:
    
    def __init__(self, fname = None, traj_data = None):
        
        # We must either have a datafile or get the data directly passed to us
        assert fname or traj_data
        
        if fname:
            fname = 'VortexEvolution_' + fname
            with open(fname, "rb") as f:
                data = pickle.load(f)                
        else:
            data = traj_data
        
        self.fname = fname
        
        self.settings = data['settings']
        self.vortices = data['vortices']
        self.trajectories = data['trajectories']
        self.circulations = data['circulations']
        
        self.conventions = PVM_Conventions()
        
        self.dipoles = []
        self.clusters = []

    
    # Run a complete cluster analysis for all time frames
    # find_dipoles must be ran prior to find_clusters; two parts of the algorithm
    def full_analysis(self):
        # Loop over time
        for i in np.arange(self.settings['n_steps']):
            # Construct configuration with id map
            cfg = get_active_vortex_cfg(self.vortices, i)
            
            # Detect dipoles and clusters
            self.dipoles.append( self.find_dipoles(cfg) )
            self.clusters.append( self.find_clusters(cfg) )
            
            # Compute energy
            self.energies.append( self.get_energy(i) )
        
        print('Cluster analysis complete')
        
        return {
                'dipoles': self.dipoles,
                'clusters': self.clusters
                }
    
    def find_clusters(self, cfg):
        pos = cfg['positions']
        circs = cfg['circulations']
        ids = cfg['ids']
        
        clusters = []
        cluster_ids = []
        
        for i, _ in enumerate(ids):
            if ids[i] in self.dipoles[-1] or ids[i] in cluster_ids:
                continue

            # Find cluster partners
            tpos = pos[i]
            tcirc = circs[i]

            _, tar_enemy_dist = self.find_nn(cfg, i, True)

            cluster = []

            for j, _ in enumerate(ids):
                # Skip if dipole or of opposite sign
                if ids[j] in self.dipoles[-1] or circs[j] != tcirc or j == i:
                    continue

                # We found a same-sign not-dipole vortex.
                # Check their distance and their distance to nearest opposite sign

                dist_friend = eucl_dist(tpos, pos[j])

                if (tar_enemy_dist < dist_friend):
                    continue

                _, dist_enemy = self.find_nn(cfg, j, True)

                if (dist_enemy < dist_friend):
                    continue

                # Friends are closer than either are to an enemy, cluster them
                if not len(cluster):
                    cluster.append(ids[i])

                cluster.append(ids[j])

            cluster_ids.extend(cluster)
            if len(cluster):
                clusters.append(cluster)

        return clusters

    """
        Rule: if two vortices are mutually nearest neighbours, we classify them as a dipole
    """
    def find_dipoles(self, cfg):
        dipoles = []
        
        circ = cfg['circulations']
        ids = cfg['ids']

        # Loop over all vortices
        for i, _ in enumerate(ids):
            # If already classified as dipole, skip
            if ids[i] in dipoles:
                continue
            
            # Find nearest neighbour of this vortex
            j ,_ = self.find_nn(cfg, i)

            # ... and nn of nn
            i2,_ = self.find_nn(cfg, j)

            # Mutual nearest neighbour found, classify as dipole if signs match
            if ((i2 == i) and (circ[i] != circ[j])):
                dipoles.append(ids[i])
                dipoles.append(ids[j])

        return dipoles

    """
    find nearest neighbour of vortex number i. if opposite is true, find only neighbour of opposite circulation
    """
    def find_nn(self, cfg, i, opposite = False):
        pos = cfg['positions']
        circ = cfg['circulations']
        ids = cfg['ids']
        
        smallest_dist = np.Inf
        nn = -1

        for j, _ in enumerate(ids):
            if j == i or (opposite and (circ[i] == circ[j])):
                continue

            dist = eucl_dist(pos[i], pos[j])
            if (dist < smallest_dist):
                smallest_dist = dist
                nn = j

        return nn, smallest_dist
    
    
    
    def angular(self, pos):
        x_vec = pos[:, 0]
        y_vec = pos[:, 1]

        return np.sum(self.circulations*(x_vec**2 + y_vec**2))
    
    """ 
    cfg is expected of form (N,2) in cartesian coordinates
    tid is needed to extract the corresp. circulations
    domain not quite isotropic, correction to shell area needed
    """
    def pair_correlation(self, cfg, tid):
        dr = 0.1
        
        # A vortex can have neighbours up to r = 2d away, if it is on one side of the domain
        bins = np.linspace(0, 2*self.domain_radius, int(2*self.domain_radius/dr) + 1)
        
        g = np.zeros_like(bins)
        # For each point r, loop over all vortices. 
        #   For each vortex, find all neighbours within a shell at distance r of thickness dr
        
        # Vortex number - possibly we should use an ensemble average to calculate <N> and rho
        N = len(cfg)
        for i, r in enumerate(bins):
            
            for j, v in enumerate(cfg):
                av = self.shell_area(v)
                ann = self.find_an(cfg, j, r, dr)
                
            g[i] = ann/av # remember to weight by circulation if desired
        
        # Do not weight individual pair correlations yet - I think we want to do this over an ensemble.
        return g[i], N, N/(np.pi*self.domain_radius**2)

    
    """
    Expects cartesian coordinates.
    Calculates the area of a shell of radius r and thickness dr centered on the vortex coordinates v
    Takes domain boundary into account, e.g. calculates only the part of the shell contained in domain
    """
    def shell_area(self, v, r, dr):
        R = self.domain_radius
        ri = np.linalg.norm(v)
        
        theta = 2*np.arccos( ( R**2 - r**2 - ri**2)/(2*r*ri) )
        
        return r*dr*(2*np.pi - theta)
    
    """ 
    Finds all neighbours  within shell at r(dr)
    """ 
    def find_an(self, cfg, i, r, dr):
        cv = cfg[i]
        
        ann = 0
        for j, v in enumerate(cfg):
            if j == i:
                # don't calculate self-distance. needless optimization
                assert(cfg[i, :] == cfg[j, :])
                
            d = self.eucl_dist(cv, v)
            
            if d > r and d < r + dr:
                ann = ann + 1
        return ann
    
    """
    Calculates the energy of the system in a marvelously inefficient but good-looking way
    """
    def get_energy(self, frame):
        H = 0
        
        for v1 in self.vortices:
            for v2 in self.vortices:
                if v1.id == v2.id:
                    continue
                
                r2 = eucl_dist(v1.get_pos(frame), v2.get_pos(frame))
                H = H - 1/(np.pi)*np.log(r2)
        
        return H
    
    """
    
    Saves analysis to file according to conventions
    
    """
    def save(self):
        fname = self.conventions.save_conventions(self.settings['max_n_vortices'], 
                                                  self.settings['total_time'], 
                                                  self.settings['annihilation_threshold'], 
                                                  self.settings['seed'],
                                                  'Analysis')
        data = {
            'dipoles': self.dipoles,
            'clusters': self.clusters
                }
        
        with open(fname, "wb") as f:
            pickle.dump(data, f)
        
    
    
if __name__ == '__main__':
    pvm = PVM_Analysis('N30_T5_ATR0.01.dat')
    pvm.full_analysis()
    pvm.save()