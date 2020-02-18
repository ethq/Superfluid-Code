# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 12:12:50 2020

@author: Zak
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
import os

from .Utilities import pol2cart, cart2pol, eucl_dist, get_active_vortices, get_active_vortex_cfg, get_vortex_by_id, hex2one
from .Vortex import Vortex
from .PlotChoice import PlotChoice
from .Conventions import Conventions


"""

Class for plotting statistical data. 

"""

class HarryPlotter:

    def __init__(self, 
                 fname = None,
                 evolution_data = None,
                 analysis_data = None,
                 ):
        
        # If file passed, we set this variable to that eventual saving is done using the same convention
        self.fname = None
        self.analysis_data = None
        self.evolution_data = None
        
        # If filename is given, read it
        if fname:
            fname_evolution = 'Datafiles/E_' + fname + '.dat'
            fname_analysis = 'Datafiles/A_' + fname + '.dat'
            
            # Attempt to open evolution file.
            af, ef = None, None
            try:
                ef = open(fname_evolution, "rb")
            except IOError:
                print(f"File {fname_evolution} not found.")
                
            try:
                af = open(fname_analysis, 'rb')
            except IOError:
                print(f"File {fname_analysis} not found.")
            
            if ef:
                evolution_data = pickle.load(ef)
                self.settings = evolution_data['settings']
                self.n_steps = self.settings['n_steps']
                self.vortices = evolution_data['vortices']
                self.trajectories = evolution_data['trajectories']
                self.circulations = evolution_data['circulations']
                ef.close()
            if af:
                analysis_data = pickle.load(af)
                self.analysis_data = analysis_data
                self.dipoles = analysis_data['dipoles']
                self.clusters = analysis_data['clusters']
                self.energies = analysis_data['energies']
                af.close()
            

        
        # Set colours
        [setattr(self, k, v) for k,v in Conventions.colour_scheme().items()]
        
        self.symbols = {
                'free_vortex': '^',
                'dipole_vortex': 'o',
                'cluster_vortex': 's'
                }
        
        
     
    """
    IMPORTANT NOTE:
        - data is expected to be served in array, one for each line object. Thus even if 
          lines == 1, data = [data]
    TODO: allow both somehow? or does that make the interface less consistent <=> worse?
    """
    def get_axe_props(self, pc):
        d = self.analysis_data
        if pc == PlotChoice.smallestDistance:
            return  {
                        'title': 'Smallest distance',
                        'xlabel': 'Frame',
                        'ylabel': 'Distance',
                        'labels': ['<Empty>'],
                        'lines': 1,
                        'data': [d['smallestDistance']]
                    }
        elif pc == PlotChoice.auto_corr_cluster:
            return  {
                         'title': 'Cluster autocorrelation',
                         'xlabel': 'Frame',
                         'ylabel': 'Autocorr',
                         'labels': ['autocorr'],
                         'lines': 1,
                          'data': [np.array(d['auto_corr_cluster'])/np.array(d['n_cluster'])]
                    }
        elif pc == PlotChoice.energyPerVortex:
            return {
                          'title': 'Energy per vortex',
                          'xlabel': 'Frame',
                          'ylabel': 'Energy',
                          'labels': ['Energy per vortex'],
                          'lines': 1
                    }
        elif pc == PlotChoice.rmsFirstVortex:
            return {
                        'title': "RMS distance for zero vortex",
                        'xlabel': 'Frame',
                        'ylabel': 'RMS distance',
                        'labels': ['RMS distance'],
                        'lines': 1,
                         'data': [d['rmsFirstVortex']]
                    }
        elif pc == PlotChoice.rmsNonDipoleNonCentered:
            return  {
                         'title': 'Noncentered nondipole RMS',
                         'xlabel': 'Frame',
                         'ylabel': 'RMS distance',
                         'labels': ['RMS distance'],
                         'lines': 1,
                         'data': [[np.sqrt(np.mean(dv)) for dv in d['rmsNonDipoleNonCentered']]]
                    }
        elif pc == PlotChoice.rmsNonDipole:
            return  {
                         'title': 'Centered nondipole RMS',
                         'xlabel': 'Frame',
                         'ylabel': 'RMS distance',
                         'labels': ['CND distance'],
                         'lines': 1,
                          'data': [[np.sqrt(np.mean(dv)) for dv in d['rmsNonDipole']]]
                    }
        elif pc == PlotChoice.rmsClusterNonCentered:
            return                     {
                         'title': 'Noncentered cluster RMS',
                         'xlabel': 'Frame',
                         'ylabel': 'RMS distance',
                         'labels': ['NCC distance'],
                         'lines': 1,
                          'data': [[np.sqrt(np.mean(dv)) for dv in d['rmsClusterNonCentered']]]
                    }
        elif pc == PlotChoice.rmsCluster:
            return                     {
                        'title': 'RMS distance in clusters',
                        'xlabel': 'Frame',
                        'ylabel': 'RMS distance',
                        'labels': ['RMS distance'],
                        'lines': 1,
                        'data': [[np.sqrt(np.mean(dv)) for dv in d['rmsCluster']]]
                    }
        elif pc == PlotChoice.numberOfVortices:
            return                     {
                        'title': 'Number of vortices',
                        'xlabel': 'Frame',
                        'ylabel': 'Count',
                        'labels': ['Total', 'Dipoles', 'Clusters', 'Free'],
                        'lines': 4,
                        'data': [np.array(d['n_total']), np.array(d['n_dipole']), np.array(d['n_cluster']), 
                                 np.array(d['n_total']) - np.array(d['n_dipole']) - np.array(d['n_cluster'])]
                    }
        elif pc == PlotChoice.dipoleMoment:
            return                     {
                        'title': 'Dipole moment',
                        'xlabel': 'Frame',
                        'ylabel': 'Deviation',
                        'labels': ['Dipole moment'],
                        'lines': 1,
                        'data': [d['dipoleMoment']]
                    }
        elif pc == PlotChoice.energyImageReal:
            return {
                        'title': 'Energy differentials (centered)',
                        'xlabel': 'Frame',
                        'ylabel': 'Differential',
                        'labels': ['Images', 'Real'],
                        'lines': 2,
                        'data': [d['energies2'][:, 1] - np.mean(d['energies2'][0, 1]), d['energies2'][:, 0] - np.mean(d['energies2'][0, 0])]
                    }
        elif pc == PlotChoice.energy:
            return {
                        'title': 'Energy deviation',
                        'xlabel': 'Frame',
                        'ylabel': 'Deviation',
                        'labels': ['Energy deviation'],
                        'lines': 1,
                        'data': [d['energies']]
                    }
        
    
    """
    
    choice:      [Array] expected to consist of PlotChoice elements
    t:           [Integer] frame at which to plot. only relevant if vortices is specified
    
    """
    def plot(self, choice, frame = 0):
        choice = PlotChoice.validate_plot_choice(choice)
        f = plt.figure()
        f.suptitle(f"Seed: {self.settings['seed']}")
        
        axes = [f.add_subplot(len(choice), 1, i) for c, i in zip(choice, 1+np.arange(len(choice)))]
        
        for i in np.arange(len(choice)):
            prop = self.get_axe_props( choice[i] )
            axes[i].set_title(prop['title'])
            axes[i].set_xlabel(prop['xlabel'])
            axes[i].set_ylabel(prop['ylabel'])
            
            for j in np.arange(prop['lines']):
                label = prop['labels'][j]
                data = prop['data'][j]
                axes[i].plot(data, label = label)
            
            if j >= 1:
                axes[i].legend()
        plt.show()
    
    
    def _plot_cfg_t(self, i, save = False):
        cfg = get_active_vortex_cfg(self.vortices, i)
        pos = cfg['positions']
        ids = cfg['ids']
        
        # If analysis data not available, plot "naked" vortices.
        if not hasattr(self, 'dipoles'):
            self.plot_cfg(pos)
            return
        
        # TODO Could actually be replaced by an indexof call:
        # we get id from dipoles/clusters, then indexof, use it for pos
        idmap = dict(zip(ids, np.linspace(0, len(ids)-1, len(ids)).astype(int)))
        
        # Dipoles/clusters at time i
        dipoles = self.dipoles[i]
        clusters = self.clusters[i]

        # Create figure
        f = plt.figure()
        ax = f.add_subplot(111, projection = 'polar')
        # ax.grid(False)

        # Plot dipoles first
        
        # Loop over dipole ids pairwise
        for id_k, id_j in zip(dipoles[0::2], dipoles[1::2]):
            k = idmap[id_k]
            j = idmap[id_j]
            
            x = np.array([pos[k][0], pos[j][0]])[:, np.newaxis]
            y = np.array([pos[k][1], pos[j][1]])[:, np.newaxis]

            r, t = cart2pol(x, y)
            
            ax.plot(t, r, color = self.dipole_colour)


        # Then plot clusters
        c_counter = 0
            
        # Loop over clusters
        for c in clusters:
            # Find the indices in the current configuration corresp. cluster ids
            cinds = [idmap[cid] for cid in c]
            cpos = pos[cinds, :]
            x, y = cpos[:,0], cpos[:,1]
            
            r, t = cart2pol(x, y)

            ax.plot(t, r, color = self.cluster_colour)

        # Merge all cluster arrays to easily check if a vortex is clustered
        cluster_ids = list(chain.from_iterable(clusters))

        # Plot vortices themselves - convenient to iterate over ids here
        
        for j, v in enumerate(self.vortices):
            # We use this to method to avoid setting line colours midway
            if not v.is_alive(i):
                continue
            
            x, y = v.get_pos(i)
            r, theta = cart2pol(x, y)

            # Plot vortices themselves
            marker = '^'

            if v.id in dipoles:
                marker = 'o'
            elif v.id in cluster_ids:
                marker = 's'

            ax.plot(theta, r, marker = marker, ls='', markeredgecolor = 'black', markeredgewidth='1', markerfacecolor = self.vortex_colours[v.circ], zorder = 1e3) 
            
        ax.set_title(f"Vortex configuration, time: {i/len(self.dipoles)}")
        
        if save:
            if not os.path.exists(f"Plots/{self.settings['seed']}"):
                os.makedirs(f"Plots/{self.settings['seed']}")
            f.savefig(f"Plots/{self.settings['seed']}/{1000*i/len(self.dipoles)}.png")
        else:
            f.show()
        
        plt.close('all')
    """
    
    TODO: Make this use analysis data, e.g. plot cluster lines etc
    Plots a vortex configuration, expected in the format spat out by get_active_vortex_cfg():
        A dictionary with keys 'positions', 'circulations' and 'ids'. 'ids' are not used and may be left empty.
    
    """
    def plot_cfg(self, cfg = None, time_ = -1, percent = -1, save = False):
        # Plot using own data:
        if not cfg:
            if percent >= 0:
                time_ = int(np.floor( percent/100*len(self.trajectories) ))
            
            # If no config passed, we should have some non-negative time
            assert time_ != -1
            print(f'Plotting configuration at time {time_}')
            self._plot_cfg_t(time_, save)
            return
        
        plt.figure()
        
        pos = cfg['positions']
        c = np.array(cfg['circulations']).astype(int)
        c = (c+1) // 2
    
        mark = ['o', 'o']
        colors = ['#88d19b', '#853128']
        
        for i, p in enumerate(pos):
            plt.plot(p[0], p[1], mark[c[i]], color = colors[c[i]])
            
        plt.tight_layout()
        plt.show()
        
        
    def save(self):
        raise NotImplementedError('HarryPlotter.save()')
    