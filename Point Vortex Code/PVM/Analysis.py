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
import pathlib

from PVM.Utilities import pol2cart, cart2pol, eucl_dist, get_active_vortices, get_active_vortex_cfg
from PVM.Vortex import Vortex

from PVM.Conventions import Conventions


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
                'T': self.T,
                'dt': self.dt,
                'n_steps': int(self.T/self.dt),
                'domain_radius': self.domain_radius,
                'annihilation_threshold': self.annihilation_threshold,
                'tol': self.tol
                } 
        
        trajectories is an (N_steps, N_vortices, 2) array.
        vortices is an array consisting of Vortex classes - dead or alive
        circulations is an (N_steps, N_vortices, N_vortices) array. 
"""
class Analysis:
    
    def __init__(self, fname = None, traj_data = None):
        
        # We must either have a datafile or get the data directly passed to us
        assert fname or traj_data
        
        if fname:
            fname = 'Datafiles/Evolution_' + fname
            with open(fname, "rb") as f:
                data = pickle.load(f)                
        else:
            data = traj_data
        
        self.fname = fname
        
        self.settings = data['settings']
        self.vortices = data['vortices']
        self.trajectories = data['trajectories']
        self.circulations = data['circulations']
        
        self.dipoles = []
        self.clusters = []
        self.energies = []
        self.dipole_moments = []
    
    # Run a complete cluster analysis for all time frames
    # find_dipoles must be run prior to find_clusters; two parts of the algorithm
    def full_analysis(self):
        st = time.time()
        print('starting analysis')
        
        # Loop over time
        for i in np.arange(self.settings['n_steps']):
            # Construct configuration with id map
            cfg = get_active_vortex_cfg(self.vortices, i)
            
            # Detect dipoles and clusters
            self.dipoles.append( self.find_dipoles(cfg) )
            self.clusters.append( self.find_clusters(cfg) )
            
            # Compute energy
            self.energies.append( self.get_energy(i) )
            
            # Compute dipole moment
            self.dipole_moments.append( self.get_dipole_moment(i) )
            
            # Compute weighted pair correlation
            
        self.energies = np.array(self.energies)
        self.dipole_moments = np.array(self.dipole_moments)
        
        tt = time.time() - st
        mins = tt // 60
        sec = tt % 60
        print('analysis complete after %d min %d sec' % (mins, sec))
        
        return {
                'dipoles': self.dipoles,
                'clusters': self.clusters,
                'energies': self.energies
                }
        
    def get_dipole_moment(self, i):
        D = 0
        
        for v in self.vortices:
            if not v.is_alive(i):
                continue
            
            D = D + v.circ*v.get_pos(i)
        
        return np.abs(D)/len(D)
    
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
    
    
    # Computes the angular mometum of the gas
    def angular(self, t):
        A = 0
        
        for v in self.vortices:
            if not v.is_alive(t):
                continue
            
            pos = v.get_pos(t)
            A = A + v.circ*np.linalg.norm(pos)**2
            
        return A
    
    """ 
    cfg is expected of form (N,2) in cartesian coordinates
    tid is needed to extract the corresp. circulations
    domain not isotropic, correction to shell area needed
    current assumption is a circular domain
    """
    def weighted_pair_correlation(self, cfg, tid):
        # Shell thickness
        dr = 0.1
        
        # A vortex can have neighbours up to r = 2d away, if it is on one side of the domain
        bins = np.linspace(0, 2*self.domain_radius, int(2*self.domain_radius/dr) + 1)
        
        circ = self.circulations[tid][0, :]
        
        assert len(circ) == len(cfg)
        
        g = np.zeros_like(bins)
        # For each point r, loop over all vortices. 
        #   For each vortex, find all neighbours within a shell at distance r of thickness dr
        
        # Vortex number - possibly we should use an ensemble average to calculate <N> and rho
        N = len(cfg)
        
        # Iterate over shells at various radi
        for i, r in enumerate(bins):
            # For a fixed shell, find the contribution from each vortex
            total = 0
            for j, v in enumerate(cfg):
                # Calculated the weighted shell area
                av = self.shell_area(v)
                
                # Find all neighbours contained in this shell
                ann = self.find_an(cfg, j, r, dr)
                
                # Weight them by their circulation and sum
                ann = np.sum([circ[k] for k in ann])
                
                total = total + ann*circ[i]
                
            g[i] = total/(av*N) # remember to weight by circulation if desired
        
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
    
    Input:
        cfg: configuration, expected in the form of an (N,2) array of cartesian coordinates
        i:   target on which the shell is centered
        r:   shell radius
        dr:  shell thickness
        
    Output:
        list containing the indices of the neighbours within the shell
    """ 
    def find_an(self, cfg, i, r, dr):
        cv = cfg[i]
        
        ann = []
        for j, v in enumerate(cfg):
            if j == i:
                # don't calculate self-distance. needless optimization
                assert(cfg[i, :] == cfg[j, :])
                
            d = self.eucl_dist(cv, v)
            
            if d > r and d < r + dr:
                ann.append(j)
        return ann
    
    
    """
    Calculates the energy of the system in a marvelously inefficient but good-looking way
    
    frame:     [Integer] used to determine the energy at any given time.
    
    stacked:   [Boolean] if true, returns the energy in a dict of type {vortex_id: vortex_energy}
                         this is useful for importance sampling when one vortex is updated at a time
                         
    vortex_id: [Integer] if specified, returns _only_ the energy for the given vortex
    """
    def get_energy(self, frame, stacked = False, vortex_id = -1):
        # Hamiltonian
        H = 0
        
        # Hamiltonian, if stacking
        H2 = {}
        
        # Only bother properly initializing if we're going to use it
        if stacked:
            for v in self.vortices:
                H2[v.id] = 0
        
        # Facilitate returning a single vortex energy
        vortex_targets = self.vortices
        
        # If it is wanted...
        if vortex_id != -1:
            vortex_targets = [v for v in self.vortices if v.id == vortex_id]
            
            if not len(vortex_targets):
                print(f'No vortex with id = {vortex_id}')
                return 0
        
        
        # Loop over target vortex energies/energy
        for v1 in vortex_targets:
            if not v1.is_alive(frame):
                continue
            
            # Loop over all other vortices
            for v2 in self.vortices:
                if not v2.is_alive(frame):
                    continue
                
                # Contribution from vortex-vortex interactions
                # Get its position
                v1p = np.array(v1.get_pos(frame))
                v2p = np.array(v2.get_pos(frame))
                r2 = eucl_dist(v1p, v2p)
                
                # Exclude self-interaction
                if v1.id != v2.id:
                    dH1 = - v1.circ*v2.circ/(np.pi)*np.log(r2)
                    
                    H = H + dH1
                
                # Contribution from vortex-image interactions
                # We take the interaction to be between vortex v1 and the image of v2, keeping "singular" term
                
                v2ip = np.array(v2.get_impos(frame, self.settings['domain_radius']))
                
                ri2 = eucl_dist(v1p, v2ip)
                
                dH2 = v1.circ*v2.circ/np.pi*np.log(ri2)
                H = H + dH2
                
                if stacked:
                    H2[v1.id] = H2[v1.id] + dH1 + dH2
        return H
    
    """
    
    Saves analysis to file according to conventions
    
    """
    def save(self):
        fname = Conventions.save_conventions(self.settings['max_n_vortices'], 
                                                  self.settings['T'], 
                                                  self.settings['annihilation_threshold'], 
                                                  self.settings['seed'],
                                                  'Analysis')
        
        path = pathlib.Path(fname)
        if path.exists() and path.is_file():
            raise ValueError('This seed has already been analyzed.')
        
        data = {
            'dipoles': self.dipoles,
            'clusters': self.clusters,
            'energies': self.energies
                }
        
        with open(fname, "wb") as f:
            pickle.dump(data, f)
        
    
    
if __name__ == '__main__':
    pvm = Analysis('N62_T5_ATR0.01.dat')
    pvm.full_analysis()
    pvm.save()