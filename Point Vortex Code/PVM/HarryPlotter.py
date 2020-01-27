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
        
        # If filename is given, read it
        if fname:
            fname_evolution = 'Datafiles/Evolution_' + fname + '.dat'
            fname_analysis = 'Datafiles/Analysis_' + fname + '.dat'
            
            ef = open(fname_evolution, "rb")
            af = open(fname_analysis, "rb")
            
            evolution_data = pickle.load(ef)
            analysis_data = pickle.load(af)
            
            # If saving animation and file is passed, make sure to save using the same convention
            self.fname = 'Animations/' + fname + '.mp4'
            
        
        
        # Otherwise, we assume data has been passed directly
        self.settings = evolution_data['settings']
        self.vortices = evolution_data['vortices']
        self.trajectories = evolution_data['trajectories']
        self.circulations = evolution_data['circulations']
        self.dipoles = analysis_data['dipoles']
        self.clusters = analysis_data['clusters']
        self.energies = analysis_data['energies']
        self.energies2 = np.array(analysis_data['energies2'])
        
        # Remembering to close the files
        if fname:
            ef.close()
            af.close()
            
        self.n_steps = self.settings['n_steps']
        
        # vortices spinning cw(ccw) are coloured black(red)
        red = (*hex2one('#383535'), 0.7)
        black = (*hex2one('#bd2b2b'), 0.7)
        self.vortex_colours = {-1: black, 1: red} # Indexed by ciculation
        self.dipole_colour = '#c0e39d'
        self.cluster_colour = '#57769c'
        
        self.symbols = {
                'free_vortex': '^',
                'dipole_vortex': 'o',
                'cluster_vortex': 's'
                }
        
        self.ax_props = {
                PlotChoice.energy: 
                    {
                        'title': 'Energy deviation',
                        'xlabel': 'Frame',
                        'ylabel': 'Deviation',
                        'labels': ['Energy deviation'],
                        'lines': 1,
                        'data': [analysis_data['energies']]
                    },
                PlotChoice.energyImageReal:
                    {
                        'title': 'Energy differentials (centered)',
                        'xlabel': 'Frame',
                        'ylabel': 'Differential',
                        'labels': ['Images', 'Real'],
                        'lines': 2,
                        'data': [self.energies2[:, 1]/np.mean(self.energies2[0, 1]), self.energies2[:, 0]/np.mean(self.energies2[0, 0])]   # (1580, 480)
                    },
                PlotChoice.dipoleMoment:
                    {
                        'title': 'Dipole moment',
                        'xlabel': 'Frame',
                        'ylabel': 'Deviation',
                        'labels': ['Dipole moment'],
                        'lines': 1
                    },
                PlotChoice.numberOfVortices:
                    {
                        'title': 'Number of vortices',
                        'xlabel': 'Frame',
                        'ylabel': 'Count',
                        'labels': ['Total', 'Dipoles', 'Clusters', 'Free'],
                        'lines': 4,
                        'data': [analysis_data['n_total'], analysis_data['n_dipole'], analysis_data['n_cluster'], 
                                 analysis_data['n_total'] - analysis_data['n_dipole'] - analysis_data['n_cluster']]
                    },
                PlotChoice.rmsCluster:
                    {
                        'title': 'RMS distance in clusters',
                        'xlabel': 'Frame',
                        'ylabel': 'RMS distance',
                        'labels': ['RMS distance'],
                        'lines': 1,
                        'data': [analysis_data['rmsCluster']]
                    },
                PlotChoice.rmsFirstVortex:
                    {
                          'title': "RMS distance for zero vortex",
                          'xlabel': 'Frame',
                          'ylabel': 'RMS distance',
                          'labels': ['RMS distance'],
                          'lines': 1,
                          'data': [analysis_data['rmsFirstVortex']]
                    },
                PlotChoice.energyPerVortex:
                    {
                          'title': 'Energy per vortex',
                          'xlabel': 'Frame',
                          'ylabel': 'Energy',
                          'labels': ['Energy per vortex'],
                          'lines': 1
                    },
                PlotChoice.smallestDistance:
                    {
                        'title': 'Smallest distance',
                        'xlabel': 'Frame',
                        'ylabel': 'Distance',
                        'labels': ['<Empty>'],
                        'lines': 1,
                        'data': [analysis_data['smallestDistance']]
                    }
                }
        
    """
    
    choice:      [Array] expected to consist of PlotChoice elements
    t:           [Integer] frame at which to plot. only relevant if vortices is specified
    
    """
    def plot(self, choice, frame = 0):
        choice = PlotChoice.validate_plot_choice(choice)
        f = plt.figure()
        
        axes = [f.add_subplot(len(choice), 1, i) for c, i in zip(choice, 1+np.arange(len(choice)))]
        
        for i in np.arange(len(choice)):
            prop = self.ax_props[choice[i]]
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
    
    """
    
    TODO: Make this use analysis data, e.g. plot cluster lines etc
    Plots a vortex configuration, expected in the format spat out by get_active_vortex_cfg():
        A dictionary with keys 'positions', 'circulations' and 'ids'. 'ids' are not used and may be left empty.
    
    """
    def plot_cfg(cfg):
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
    