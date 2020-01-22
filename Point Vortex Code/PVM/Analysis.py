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
        self.w_pair_corr = []
        self.rms_dist = []
    
    # Run a complete cluster analysis for all time frames
    # find_dipoles must be run prior to find_clusters; two parts of the algorithm
    # TOOD add analysis options, we may certainly not want to compute -everything-
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
            
            # Compute RMS distance
            self.rms_dist.append( self.get_rms_dist(i) )
            
            # Compute energy
            self.energies.append( self.get_energy(i) )
            
            # Compute dipole moment
            self.dipole_moments.append( self.get_dipole_moment(i) )
            
            # Compute weighted pair correlation
            self.w_pair_corr.append( self.get_pair_corr3(cfg) )
            
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
            
            D = D + np.sign(v.circ)*v.get_pos(i)
        
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
    find nearest neighbour of vortex number i. 
    if opposite is true, find only neighbour of opposite circulation
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
    
    
    # Computes the angular mometum of the gas, or if you will - the second moment of vorticity.
    # (it should be conserved - at least under non-dissipative evolution)
    def angular(self, t):
        A = 0
        
        for v in self.vortices:
            if not v.is_alive(t):
                continue
            
            pos = v.get_pos(t)
            A = A + v.circ*np.dot(pos, pos)
            
        return A
    
    """
    
    Calculates the pair correlation function for point vortices in a disk.
    
    - All vortices contribute to the statistics, even those near the boundary. 
    
    - If it is sufficient to use particles not near the boundary, get_pair_corr2 is much faster.
    
    - If weighted is set to true, the contribution of each vortex pair is weighted by their circulations
    
    Input:
        cfg:   [Dictionary] Expected keys: positions & circulations. 
                            positions: [Numpy array] expected shape: (N, 2) for N vortices
                            circulations: [Numpy array] expected shape: (N, 1) for N vortices
                            It is furthermore expected that these arrays match.
        rmax:  [Float] Maximal value of g(r)
        weighted: [Bool] Whether to weight g(r) by vortex circulations
        dr:    [Float] Increment at which g(r) is calculated
    Output:
        g(r) [Array] Correlation function
        r    [Array] Radial coordinate
    """
    def get_pair_corr3(self, cfg, rmax = 5, weighted = False, dr = .1):
        circs = cfg['circulations']
        cfg = cfg['positions']
        
        # Domain radius
        R = self.settings['domain_radius']
        
        # Total number of vortices
        N = len(cfg)
        
        # Density
        density = N/(np.pi*R**2)        
        
        # Radial bins. 
        edges = np.arange(0, rmax + 1.1 * dr, dr)
        incr = len(edges) - 1
        
        # Getting the bins as calculated by numpy
        radi = np.zeros(incr)
        for i in np.arange(incr):
            radi[i] = (edges[i] + edges[i+1])/2
        
        
        # If weighted, we bin with circulation and add at the end
        if weighted:
            edges = np.hstack((-1*np.flip(edges)[:-1], edges))

        # Pair correlation function, to be averaged over vortices
        g = np.zeros([N, incr])
        
        # Loop over all vortices
        for i, p in enumerate(cfg):
            # Distance to all other vortices
            dist = np.linalg.norm(cfg - p, axis = 1)
            
            # Weight if desired
            if weighted:
                dist = dist*circs
            
            # Kill self-contribution
            dist = dist[dist != 0]
            
            # Bin according to radiuseses, each on a dr increment
            (result, bins) = np.histogram(dist, bins = edges, normed = False)
            
#            plt.hist(dist, bins = edges, normed = False)
#            plt.show()
            # For each radius for the given vortex, calculate the effective shell area
            areas = np.array([self.shell_area(p, r, dr) for r in radi])
            
            # If weighted, sum negatives and positives to net result
            if weighted:
                ns = np.flip(result[:len(result)//2])*-1
                ps = result[len(result)//2:]
    
                result = ps + ns
            
            # Divide by effective density
            g[i, :] = result / areas
        
        
        # Average over vorticies to get g(r)
        gavg = np.zeros(incr)
        for i in np.arange(incr):          
            gavg[i] = np.mean(g[:, i])/density
            
        return gavg, radi
    
    def get_pair_corr2(self, cfg):
        dr = .1
        rmax = 5
        density = len(cfg)/(np.pi*20**2)
        mask = np.linalg.norm(cfg-np.array([10,10]), axis = 1)
        mask = mask + rmax < 10
        cfg2 = cfg[mask]
        print(len(cfg2))
        
        
        edges = np.arange(0., rmax + 1.1 * dr, dr)
        incr = len(edges) - 1
        
        g = np.zeros([len(cfg2), incr])
        
        for i, p in enumerate(cfg2):
            dist = np.linalg.norm(cfg-p, axis = 1)
            dist = dist[dist != 0]
            
            (result, bins) = np.histogram(dist, bins = edges, normed = False)
            
            g[i, :] = result / density
            
        radi = np.zeros(incr)
        
        gavg = np.zeros(incr)
        for i in np.arange(incr):
            radi[i] = (edges[i] + edges[i+1])/2
            rout = edges[i+1]
            rin = edges[i]
            
            gavg[i] = np.mean(g[:, i])/(np.pi*(rout**2-rin**2))
            
        return gavg, radi
            
    
    """
    
    Calculates the root mean square displacement of vortices.
    Cluster analysis must be done prior to calling this function.
    
    cfg:      [Dictionary] Expects keys 'positions', 'circulations' & 'ids'
                           positions: [Numpy array] Expects shape (N, 2) for N vortices
                           circulations:[Numpy array] Expects shape (N, 1) for N vortices
                           ids:         [Numpy array] Expects shape (N, 1) for N vortices
    
    dipoles:  [Array]:    Contains IDs of vortices which are classified as dipoles
    
    clusters: [Array]:    Contains IDs of vortices which are classified as clusters
    
    which:    [Integer]    Can take values in [0, 4]
                           0:   Count only clusters
                           1:   Count only dipoles
                           2:   Count only free 
                           3:   Count only non-dipoles
                           4:   Count only non-clusters
    
    """
    
    def get_rms_dist(self, tid, which = 0):
        # We assume a cluster analysis has been performed. For now, assert out if not
        assert tid < len(self.dipoles) and tid < len(self.clusters)
        
        # Put together the ids which we compute rms on
        ids = np.array([])
        
        if which == 0:
            ids = np.array(self.clusters[tid])
        elif which == 1:
            ids = np.array(self.dipoles[tid])
        # TODO implement the rest
        
        if not len(ids):
            return 0
        
        ids = np.concatenate(ids)
        
        # Get the living and matching vortices
        mask = [v.is_alive(tid) and v.id in ids for v in self.vortices]
#        mask1 = [v.is_alive(tid) for v in self.vortices]
#        mask2 = [v.id in ids for v in self.vortices]
#        mask3 = mask1 and mask2
#        print(ids)
        v = self.vortices[mask]
        
        # Number of vortices we count
        N0 = len(v)
        
        if not N0:
            return 0
        
        # Calculate the RMS for them
        srms = [np.linalg.norm(v1.get_pos(tid) - v1.get_pos(0))**2 for v1 in v]
        

        
        rms = np.sqrt(1/N0*np.sum(srms))
        
        return rms
    
    """ 
    cfg is expected of form (N, 2) in cartesian coordinates
    tid is needed to extract the corresp. circulations
    domain not isotropic, correction to shell area needed
    current assumption is a circular domain
    """
    def get_pair_corr(self, cfg, tid = 0):
        # Shell thickness
        dr = 0.1
        rmax = 1
        
        # A vortex can have neighbours up to r = 2d away, if it is on one side of the domain
        bins = dr + np.arange(0, rmax + 1.1*dr, dr)
        
        circ = self.circulations[tid][0, :]
        
        assert len(circ) == len(cfg)
        
        g = np.zeros_like(bins)
        # For each point r, loop over all vortices. 
        # For each vortex, find all neighbours within a shell at distance r of thickness dr
        
        # Vortex number - possibly we should use an ensemble average to calculate <N> and rho
        N = len(cfg)
        density = N/(np.pi*self.settings['domain_radius']**2)
        
        # Iterate over shells at various radi
        for i, r in enumerate(tqdm(bins)):
            # For a fixed shell, find the contribution from each vortex
            for j, v in enumerate(cfg):
                # Calculated the weighted shell area
                av = self.shell_area(v, r, dr)
                
                # Find all neighbours contained in this shell
                ann = self.find_an(cfg, j, r, dr)
                
                # Weight them by their circulation and area and sum
#                ann = np.sum([circ[k] for k in ann])/av
                ann = np.sum(ann)/av
#                if i == 0:
#                    print(ann, av)
                g[i] = g[i] + ann
            g[i] = np.mean(g[i])/density
        
        return g, bins

    
    """
    Expects cartesian coordinates.
    Calculates the area of a shell of radius r and thickness dr centered on the vortex coordinates v
    Takes domain boundary into account, e.g. calculates only the part of the shell contained in domain
    """
    def shell_area(self, v, r, dr):
        R = self.settings['domain_radius']
        ri = np.linalg.norm(v)
        
        # If the shell surrounding the vortex v is contained in the domain, just return the full shell area
        if r+ri < R:
            return 2*np.pi*r*dr

        # If not, get the angle "missed" as the shell intersects the domain boundary
        x = ( R**2 - r**2 - ri**2)/(2*r*ri)
        
        theta = 2*np.arccos(x)
        
        # And return the shell area without the part that hits the boundary
        return r*dr*(2*np.pi - theta)
    
    """ 
    Finds all neighbours  within shell at r(dr)
    
    Input:
        cfg: configuration, expected in the form of an (N,2) array of cartesian coordinates
        i:   target on which the shell is centered
        r:   shell radius
        dr:  shell thickness
        
    Output:
        list containing a mask of neighbours
    """ 
    def find_an(self, cfg, i, r, dr):
        cv = cfg[i]
        
        ann = []
        for j, v in enumerate(cfg):
            if j == i:
                # don't calculate self-distance. 
                ann.append(0)
                continue
                
            d = eucl_dist(cv, v)
            
            if d > r and d <= r + dr:
                ann.append(1)
            else:
                ann.append(0)
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
                # Note: np.log will give a runtimewarning if r2 == 0, but will correctly spit out -infty
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
                    H2[v1.id] = H2[v1.id] + dH2
                    if v1.id != v2.id:
                        H2[v1.id] = H2[v1.id] + dH1
            if stacked:
                return H2
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