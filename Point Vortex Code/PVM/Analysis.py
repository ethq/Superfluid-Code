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

from PVM.Utilities import pol2cart, cart2pol, eucl_dist, get_active_vortices, get_active_vortex_cfg, merge_common
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

class RMS_CHOICE:
    CLUSTER = 1
    DIPOLE = 2
    FREE = 3
    ALL = 4
    FIRST_VORTEX = 5
    NON_DIPOLE = 6
    
    REL_INITIAL_POS = 998
    REL_ORIGIN = 999
    
### Note when adding: make _SURE_ the values match up to the ones given in get_data()  
class ANALYSIS_CHOICE:
    FULL = 'full',
    
    RMS_NON_DIPOLE_CENTERED = 'rmsNonDipole'
    RMS_NON_DIPOLE_NON_CENTERED = 'rmsNonDipoleNonCentered'
    RMS_CLUSTER_CENTERED = 'rmsCluster'
    RMS_CLUSTER_NON_CENTERED = 'rmsClusterNonCentered'
    RMS_FIRST_VORTEX = 'rmsFirstVortex'

    DIPOLE_MOMENT = 'dipole_moment'
    ENERGIES = 'energies'
    
    PAIR_CORR = 'pair_corr'
    PAIR_CORR_W = 'pair_corr_w'
    AUTO_CORR = 'auto_corr'
    AUTO_CORR_CLUSTER = 'auto_corr_cluster'
    AUTO_CORR_NON_DIPOLE = 'auto_corr_nondipole'
    AUTO_CORR_DIPOLE = 'auto_corr_dipole'
    
    CLUSTER_ANALYSIS = 'cluster_analysis'

class Analysis:
    
    def __init__(self, fname = None, traj_data = None, options = {
            'frameskip': 1
            }):
        
        if fname:
            fname = 'Datafiles/E_' + fname + '.dat'
            with open(fname, "rb") as f:
                data = pickle.load(f)                
        else:
            data = traj_data
        
        # Allow this for extending
        if not fname and not traj_data:
            data = {
                'settings': None,
                'vortices': None,
                'trajectories': None,
                'circulations': None
                }
        

        
        self.fname = fname
            
        self.options = options
        self.settings = data['settings']
        self.vortices = data['vortices']
        self.trajectories = data['trajectories']
        self.circulations = data['circulations']
        
        self.dipoles = []
        self.clusters = []
        self.energies = []
        self.energies2 = []
        self.dipole_moment = []
        self.pair_corr_w = []
        self.pair_corr = []
        self.rmsCluster = []
        self.rmsNonDipole = []
        self.rmsClusterNonCentered = []
        self.rmsNonDipoleNonCentered = []
        self.rmsFirstVortex = []
        self.smallestDistance = []
        
        self.auto_corr = []
        self.auto_corr_cluster = []
        self.auto_corr_nondipole = []
        self.auto_corr_dipole = []
        
        self.n_total = []
        self.n_dipole = []
        self.n_cluster = []
    
    
    # Calculates the properties given by props. debug props must be given manually
    # props is expected to be a list of values enumerated by ANALYSIS_CHOICE
    def run(self, props = []):
        if not props:
            return
        
        cluster = False
        # Are we doing cluster analysis?
        if ANALYSIS_CHOICE.FULL in props:
            cluster = True
            props.remove(ANALYSIS_CHOICE.FULL)
        if ANALYSIS_CHOICE.CLUSTER_ANALYSIS in props:
            cluster = True
            props.remove(ANALYSIS_CHOICE.CLUSTER_ANALYSIS)
        
        st = time.time()
        print('starting analysis')

        for t in tqdm(np.arange(self.settings['n_steps'])):
            if t % self.options['frameskip']:
                continue
            
            cfg = get_active_vortex_cfg(self.vortices, t)
            
            # Do complete cluster analysis if desired
            if cluster:
                # Detect dipoles and clusters
                self.dipoles.append( self.find_dipoles(cfg) )
                self.clusters.append( self.find_clusters(cfg) )
                
                # Vortex number statistics
                self.n_total.append( len(cfg['ids']) ) 
                self.n_dipole.append( len(self.dipoles[-1]) ) 
                self.n_cluster.append( len(np.unique(self.clusters[-1])) )
            
            
            # Loop over properties to analyze
            for p in props:
                # Get the function that does the udpating
                f = 'get_' + p
                # print(f,p)
                # Make sure we aint doin stupid stuff
                assert hasattr(self, f)
                assert hasattr(self, p)
                
                # Call it
                val = getattr(self, f)(t, cfg)
                
                # Add it
                getattr(self, p).append(val)
        
        for p in props:
            v = getattr(self, p)
            setattr(self, p, np.array(v))
        
    
    # Run a complete cluster analysis for all time frames
    # find_dipoles must be run prior to find_clusters; two parts of the algorithm
    # TOOD add analysis options, we may certainly not want to compute -everything-, 
    # Or on every frame...
    def full_analysis(self):
        # Here we just take all flags & call run() with em.            
        props = list(self.get_data().keys())
        
        # Remove stuff related to cluster analysis
        props.remove('dipoles')
        props.remove('clusters')
        props.remove('n_total')
        props.remove('n_dipole')
        props.remove('n_cluster')
        
        # And add main cluster flag
        props.append(ANALYSIS_CHOICE.CLUSTER_ANALYSIS)
        
        # Analyze
        self.run(props)
        
        # Leave the old code for now...
        return self.get_data()
        
        # # Loop over time
        # for i in np.arange(self.settings['n_steps']):
        #     if i % self.options['frameskip']:
        #         continue
            
        #     # Construct configuration with id map
        #     cfg = get_active_vortex_cfg(self.vortices, i)
            
        #     # Detect dipoles and clusters
        #     self.dipoles.append( self.find_dipoles(cfg) )
        #     self.clusters.append( self.find_clusters(cfg) )
            
        #     # Vortex number statistics
        #     self.n_total.append( len(cfg['ids']) ) 
        #     self.n_dipole.append( len(self.dipoles[-1]) ) 
        #     self.n_cluster.append( len(np.unique(self.clusters[-1])) )
            
        #     # Compute smallest distance(does this cause problems with image energies..???)
        #     # self.smallestDistance.append( self.get_smallest_distance(i) )
            
        #     # Compute RMS distance for non-dipoles
        #     self.rmsNonDipole.append( self.get_rms_dist(i, which = RMS_CHOICE.NON_DIPOLE) )
            
        #     # Compute RMS distance for non-dipoles, the Barenghi way
        #     self.rmsNonDipoleNonCentered.append( self.get_rms_dist(i, which = RMS_CHOICE.NON_DIPOLE, rel = RMS_CHOICE.REL_INITIAL_POS))
            
        #     # Compute RMS distance of clusters, the Barenghi way
        #     self.rmsClusterNonCentered.append( self.get_rms_dist(i, which = RMS_CHOICE.CLUSTER, rel = RMS_CHOICE.REL_INITIAL_POS) )
            
        #     # Compute RMS distance
        #     self.rmsCluster.append( self.get_rms_dist(i, which = RMS_CHOICE.CLUSTER) )
            
        #     # Compute RMS distance of vortex id = 0
        #     # self.rmsFirstVortex.append( self.get_rms_dist(i, which = RMS_CHOICE.FIRST_VORTEX ) )
            
        #     # Compute the temporal autocorelation
        #     self.auto_corr.append( self.get_auto_corr(i) )
            
        #     # Compute energy
        #     # self.energies.append( self.get_energy(i) )
            
        #     # For debug purposes, compute energy from images and real vortices separately
        #     # self.energies2.append( self.get_energy(i, debug = True))
            
        #     # Compute dipole moment
        #     # self.dipole_moment.append( self.get_dipole_moment(i) )
            
        #     # Compute weighted pair correlation
        #     # self.pair_corr_w.append( self.get_pair_corr3(cfg, weighted = True) )
            
        #     # Compute non-weighted pair correlation
        #     # self.pair_corr.append( self.get_pair_corr3(cfg) )
            
        # # All should be turned into numpy arrays
        # self.energies = np.array(self.energies)
        # self.dipole_moment = np.array(self.dipole_moment)
        
        # tt = time.time() - st
        # mins = tt // 60
        # sec = tt % 60
        # print('analysis complete after %d min %d sec' % (mins, sec))
        
        # return self.get_data()
    
    
    def get_energies(self, t, cfg = None):
        return self.get_energy(t)
    
    def get_energies2(self, t, cfg = None):
        return self.get_energy(t, debug = True)

    def get_rmsCluster(self, t, cfg = None):
        return self.get_rms_dist(t, which = RMS_CHOICE.CLUSTER)
    
    def get_rmsNonDipole(self, t, cfg = None):
        return self.get_rms_dist(t, which = RMS_CHOICE.NON_DIPOLE)
    
    def get_rmsClusterNonCentered(self, t, cfg = None):
        return self.get_rms_dist(t, which = RMS_CHOICE.CLUSTER, rel = RMS_CHOICE.REL_INITIAL_POS)
    
    def get_rmsNonDipoleNonCentered(self, t, cfg = None):
        return self.get_rms_dist(t, which = RMS_CHOICE.NON_DIPOLE, rel = RMS_CHOICE.REL_INITIAL_POS)
    
    def get_rmsFirstVortex(self, t, cfg = None):
        return self.get_rms_dist(t, which = RMS_CHOICE.FIRST_VORTEX )
    
    def get_pair_corr(self, t, cfg):
        return self.get_pair_corr3(cfg)
    
    def get_pair_corr_w(self, t, cfg):
        return self.get_pair_corr3(cfg, weighted = True)
    
    def get_auto_corr(self, t, cfg):
        return self.get_autocorrelation(t)
    
    def get_auto_corr_cluster(self, t, cfg):
        return self.get_autocorrelation(t, which = RMS_CHOICE.CLUSTER)
    
    def get_auto_corr_nondipole(self, t, cfg):
        return self.get_autocorrelation(t, which = RMS_CHOICE.NON_DIPOLE)
    
    def get_auto_corr_dipole(self, t, cfg):
        return self.get_autocorrelation(t, which = RMS_CHOICE.DIPOLE)
    
    def get_dipole_moment(self, t, cfg):
        D = 0
        
        for v in self.vortices:
            if not v.is_alive(t):
                continue
            
            D = D + np.sign(v.circ)*v.get_pos(t)
        
        return np.abs(D)/len(D)
        
    
    # Set to true to save debug data, such as im/real energies, smallest dist etc
    def get_data(self, debug = False):
        public = {
            'dipoles': self.dipoles,
            'clusters': self.clusters,
            'n_total': self.n_total,
            'n_dipole': self.n_dipole,
            'n_cluster': self.n_cluster,
            'energies': self.energies,
            # 'energies2': self.energies2,
            'dipole_moment': self.dipole_moment,
            'rmsCluster': self.rmsCluster,
            'rmsNonDipole': self.rmsNonDipole,
            'rmsClusterNonCentered': self.rmsClusterNonCentered,
            'rmsNonDipoleNonCentered': self.rmsNonDipoleNonCentered,
            'rmsFirstVortex': self.rmsFirstVortex,
            'pair_corr': self.pair_corr,
            'pair_corr_w': self.pair_corr_w,
            'auto_corr': self.auto_corr,
            'auto_corr_cluster': self.auto_corr_cluster,
            'auto_corr_nondipole': self.auto_corr_nondipole,
            'auto_corr_dipole': self.auto_corr_dipole
            # 'smallestDistance': self.smallestDistance
        }
    
        private = {
            'energies2': self.energies2,
            'smallestDistance': self.smallestDistance
        }
        
        if debug:
            public.update(private)
            
        return public
        
    
    # Completes a given file to a full analysis. 
    # Note: only fills in _empty_ attributes, if some attribute has been calculated using old code it remains
    
    # Note2: if props is non-empty, only extends with attributes contained therein
    def extend(self, f, props = []):
        fname_analysis = 'Datafiles/Analysis_' + f + '.dat'    
        fname_evolution = 'Datafiles/Evolution_' + f + '.dat'
        
        ef = open(fname_evolution, "rb")
        af = open(fname_analysis, "rb")
                
        evolution_data = pickle.load(ef)
        analysis_data = pickle.load(af)
        
        self.fname = f
        self.settings = evolution_data['settings']
        self.vortices = evolution_data['vortices']
        self.trajectories = evolution_data['trajectories']
        self.circulations = evolution_data['circulations']

        # Get the properties that are exposed to users of this class, or those who load its saved files        
        sk = self.get_data().keys()
        
        # Set only these properties from analysis_data
        [setattr(self, k, v) for k, v in analysis_data.items() if k in sk]

        # Remember remember the fifth of november
        af.close()
        ef.close()
        
        to_analyze = []
        # Loop over own properties
        for attr in sk:
            # If it's set, move on
            if getattr(self, attr) or (props and attr not in props):
                continue
            
            # Otherwise, analyze and set it
            if attr in ['dipoles', 'clusters', 'n_total', 'n_dipole', 'n_cluster']:
                to_analyze.append(ANALYSIS_CHOICE.CLUSTER_ANALYSIS)
                
            else:
                to_analyze.append( attr )
        
        print(f'Analyzing: {to_analyze}')
        
        
        # Do the analysis
        self.run(to_analyze)
        
        # Re-save the file
        # self.save()
        
            
    ### remove ### --- used for debug purposes
    # def get_smallest_distance(self, i):
    #     mindist = np.Inf
    #     for v1 in self.vortices:
    #         if not v1.is_alive(i):
    #             continue
            
    #         for v2 in self.vortices:
    #             if not v2.is_alive(i):
    #                 continue
                
    #             if v1.id == v2.id:
    #                 continue
                
    #             newdist = eucl_dist(v1.get_pos(i), v2.get_pos(i))
                
    #             if newdist < mindist:
    #                 mindist = newdist
                    
    #     return mindist
            
        

    
    
    """
    This routine locates clusters _only_. It assumes dipoles have been found first.
    The typical rule for clustering is this:
        
        If two same-signed vortices are closer than either is to an opposite signed vortex,
        they belong to the same cluster.
        
    It is very important that they belong _to the same cluster_. In particular, denoting clustering by ~, 
    we may have a ~ b and b ~ c. Without explicitly putting them in the same cluster, a !~ c.
    
    """
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
        
        # Merge clusters with at least one vortex in common
        return merge_common(clusters)

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
    
    def get_autocorrelation(self, t, which = RMS_CHOICE.ALL):
        # Put together the ids which we compute rms on
        ids = np.array([])
        all_ids = [v.id for v in self.vortices]
        
        if which == RMS_CHOICE.CLUSTER:
            ids = list(chain.from_iterable(np.array(self.clusters[t])))
        elif which == RMS_CHOICE.DIPOLE:
            ids = np.array(self.dipoles[t])
        elif which == RMS_CHOICE.FIRST_VORTEX:
            ids = np.array([[0]])
        elif which == RMS_CHOICE.NON_DIPOLE:
            dipoles = self.dipoles[t]
            ids = [id_ for id_ in all_ids if id_ not in dipoles]
        elif which == RMS_CHOICE.ALL:
            ids = all_ids
        
        if not len(ids):
            return 0
        
        # print(f"Cluster ids: {cids}, Nondipole ids: {nids}")
        
        ac = np.sum(
            [np.dot(v1.get_pos(t), v1.get_pos(0)) for v1 in self.vortices if v1.is_alive(t) and v1.id in ids]
            )
        
        return ac
    
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
    
    which:    [Integer]    Enumerated in RMS_CHOICE class.
    
    
    Returns:
        Tuple containing (rms, id) where rms(id) is a length N array for N vortices
    
    """
    
    def get_rms_dist(self, tid, which = RMS_CHOICE.CLUSTER, rel = RMS_CHOICE.REL_ORIGIN):
        # We assume a cluster analysis has been performed. For now, assert out if not
        assert tid < len(self.dipoles) and tid < len(self.clusters)
        
        # Put together the ids which we compute rms on
        ids = np.array([])
        all_ids = [v.id for v in self.vortices]
        
        if which == RMS_CHOICE.CLUSTER:
            ids = list(chain.from_iterable(np.array(self.clusters[tid])))
        elif which == RMS_CHOICE.DIPOLE:
            ids = np.array(self.dipoles[tid])
        elif which == RMS_CHOICE.FIRST_VORTEX:
            ids = np.array([[0]])
        elif which == RMS_CHOICE.NON_DIPOLE:
            dipoles = self.dipoles[tid]
            ids = [id_ for id_ in all_ids if id_ not in dipoles]
        elif which == RMS_CHOICE.ALL:
            ids = all_ids
        
        if not len(ids):
            return 0
        
        # Since the id arrays are of type [[id1, id2], [id3, id4], ...], we concatenate
        # ids = np.concatenate(ids)
        
        # Get the living and matching vortices
        mask = [v.is_alive(tid) and v.id in ids for v in self.vortices]
        v = self.vortices[mask]
        
        # Number of vortices we count
        N0 = len(v)
        
        if not N0:
            return 0
        
        # Calculate the RMS for them
        # Note: there is no point doing this in a loop
        # |traj-initial_pos|^2 = rms
        
        if rel == RMS_CHOICE.REL_ORIGIN:
            srms = [np.linalg.norm(v1.get_pos(tid))**2 for v1 in v]
        elif rel == RMS_CHOICE.REL_INITIAL_POS:
            srms = [np.linalg.norm(v1.get_pos(tid) - v1.get_pos(0))**2 for v1 in v]
        return srms
    
    """ 
    cfg is expected of form (N, 2) in cartesian coordinates
    tid is needed to extract the corresp. circulations
    domain not isotropic, correction to shell area needed
    current assumption is a circular domain
    """
    def get_pair_corr4(self, cfg, tid = 0):
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
        
        # If the shell is _outside_ the domain, return infinite shell-area
        # so that we kill the contribution to g(r) there
        if np.abs(ri - r) >= R:
            return np.Inf

        # If not, get the angle "missed" as the shell intersects the domain boundary
        x = ( R**2 - r**2 - ri**2)/(2*r*ri)
        
        theta = 2*np.arccos(x)
        
#        with np.warnings.catch_warnings():
#            np.warnings.filterwarnings('error')
#            try:
#                theta = 2*np.arccos(x)
##                np.warnings.warn(Warning())
#            except Warning: 
#                print(f"The shitty values are: R: {R}, r: {r}, ri: {ri}, x: {x}, v: {v}")
#        
        
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
    def get_energy(self, frame, stacked = False, vortex_id = -1, debug = False):
        # Shortcut for radius
        R = self.settings['domain_radius']
        
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
        
        # Differential for real energy increments
        dhr = 0
        # Differential for images
        dhi = 0
        
        # This is overcounting!!! Check e.g. that 12 and 21 are counted. 
        # Just results in a multiplicative factor, but anyway.
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
                    dH1 = - v1.circ*v2.circ/np.pi*np.log(r2)
                    H = H + dH1
                    dhr = dhr + dH1
                
                # Contribution from vortex-image interactions
                # We take the interaction to be between vortex v1 and the image of v2, keeping "singular" term
                
                v2ip = np.array(v2.get_impos(frame, self.settings['domain_radius']))
                
                ri2 = np.sqrt(eucl_dist(v1p, v2ip))
                
                # Note the plus compared to minus for real vortices. This compensates for the fact that images have opposite sign.
                # The factor 2 in front is due to the fact that we calculate the real energies using r2, e.g. the squared distance.
                # Hence we must scale the image energies accordingly
                dH2 = 2*v1.circ*v2.circ/np.pi*np.log(ri2*np.linalg.norm(v2p))

#                # Same, just expanded. https://cims.nyu.edu/~obuhler/StatMechVort/BuhlerPF02.pdf
#                dH2 = v1.circ*v2.circ/np.pi*np.log(R**4 - 2*R**2*np.dot(v1p, v2p) + np.dot(v1p, v1p)*np.dot(v2p, v2p))
                
                H = H + dH2
                dhi = dhi + dH2
                
                if stacked:
                    H2[v1.id] = H2[v1.id] + dH2
                    if v1.id != v2.id:
                        H2[v1.id] = H2[v1.id] + dH1
            
            if stacked:
                return H2
            
        if debug:
            return [dhr, dhi]
        
        return H
    
    """
    
    Saves analysis to file according to conventions
    
    """
    def save(self):
        fname = Conventions.save_conventions(self.settings['max_n_vortices'], 
                                                  self.settings['T'], 
                                                  self.settings['annihilation_threshold'], 
                                                  self.settings['seed'],
                                                  domain_radius = self.settings['domain_radius'],
                                                  gamma = self.settings['gamma'],
                                                  conv = 'fresh',
                                                  data_type = 'Analysis')
        
        # path = pathlib.Path(fname)
        # resave = ''
#         FOR NOW JUST OVERWRITE
#        if path.exists() and path.is_file():
#            resave = input("The file has already been analyzed. Do you wish to overwrite? [y/n]")
#            
#            if resave == 'y':
#                pass
#            elif resave == 'no':
#                return
#            else:
#                print('Invalid input. Press [y] for yes, [n] for no. Aborting...')
#                return
        
        # TODO: standardize to PlotChoice alternatives
        data = self.get_data( debug = False )
        
        with open(fname, "wb") as f:
            pickle.dump(data, f, protocol = pickle.HIGHEST_PROTOCOL)
        
        # if resave != '':
            # print('New analysis saved.')
    
    
if __name__ == '__main__':
    pvm = Analysis('N62_T5_ATR0.01.dat')
    pvm.full_analysis()
    pvm.save()