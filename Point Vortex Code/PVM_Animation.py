# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 10:28:57 2019

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

from Utilities import pol2cart, cart2pol, eucl_dist, get_active_vortices, get_active_vortex_cfg, get_vortex_by_id
from Vortex import Vortex


class PVM_Animation:
    
    def __init__(self, fname_param = None, evolution_data = None, analysis_data = None):
        # If filename is given, read it
        if fname_param:
            fname_evolution = 'VortexEvolution_' + fname_param
            fname_analysis = 'VortexAnalysis_' + fname_param
            
            ef = open(fname_evolution, "rb")
            af = open(fname_analysis, "rb")
            
            evolution_data = pickle.load(ef)
            analysis_data = pickle.load(af)
        
        # Otherwise, we assume data has been passed directly
        self.settings = evolution_data['settings']
        self.vortices = evolution_data['vortices']
        self.trajectories = evolution_data['trajectories']
        self.circulations = evolution_data['circulations']
        self.dipoles = analysis_data['dipoles']
        self.clusters = analysis_data['clusters']
        
        # Remembering to close the files
        if fname_param:
            ef.close()
            af.close()
            
        self.n_steps = self.settings['n_steps']
            
        self.ani = None                    # For animation, handle required to stop 'on command'
        self.vortex_lines = []
        self.dipole_lines = []
        self.cluster_lines = []
        self.trail_lines = []
        
        # vortices spinning cw(ccw) are coloured black(red)
        self.vortex_colours = {-1:'#383535', 1:'#bd2b2b'} # Indexed by ciculation
        self.dipole_colour = '#c0e39d'
        self.cluster_colour = '#57769c'
        
        # Length of vortex trails in animation
        self.trail_length = 40
        
        self.symbols = {
                'free_vortex': '^',
                'dipole_vortex': 'o',
                'cluster_vortex': 's'
                }
        
    def animate_trajectories_update(self, i):
        # Are we done?
        if i >= self.n_steps:
            self.ani.event_source.stop()
            return
        
        cfg = get_active_vortex_cfg(self.vortices, i)
        pos = cfg['positions']
        ids = cfg['ids']
        
        idmap = dict(zip(ids, np.linspace(0, len(ids)-1, len(ids)).astype(int)))
        
        dipoles = self.dipoles[i]
        clusters = self.clusters[i]

        # Plot dipoles first
        for dl in self.dipole_lines:
            dl.set_xdata([])
            dl.set_ydata([])

        d_counter = 0
        for id_k, id_j in zip(dipoles[0::2], dipoles[1::2]):
            k = idmap[id_k]
            j = idmap[id_j]
            
            x = np.array([pos[k][0], pos[j][0]])[:, np.newaxis]
            y = np.array([pos[k][1], pos[j][1]])[:, np.newaxis]

            r, t = cart2pol(x, y)
            self.dipole_lines[d_counter].set_xdata(t)
            self.dipole_lines[d_counter].set_ydata(r)
            d_counter = d_counter + 1


        # Then plot clusters
        c_counter = 0

        # Clear old lines
        for cl in self.cluster_lines:
            cl.set_xdata([])
            cl.set_ydata([])
            
        # Loop over clusters
        for c in clusters:
            # Find the indices in the current configuration corresp. cluster ids
            cinds = [idmap[cid] for cid in c]
            cpos = pos[cinds, :]
            x, y = cpos[:,0], cpos[:,1]
            
            r, t = cart2pol(x, y)

            self.cluster_lines[c_counter].set_xdata(t)
            self.cluster_lines[c_counter].set_ydata(r)

            c_counter = c_counter + 1

        cluster_ids = list(chain.from_iterable(clusters))

        # Clear out trail lines
        for tr in self.trail_lines:
            tr.set_segments([])


        # Plot vortices themselves - convenient to iterate over ids here
        for vl in self.vortex_lines:
            vl.set_xdata([])
            vl.set_ydata([])
                
        living_mask = [v.is_alive(i) for v in self.vortices]
        a_vortices = self.vortices[living_mask]
        
        for j, v in enumerate(a_vortices):
            x, y = v.get_pos(i)
            r, theta = cart2pol(x, y)

            # Plot vortices themselves
            marker = '^'

            if v.id in dipoles:
                marker = 'o'
            elif v.id in cluster_ids:
                marker = 's'

            self.vortex_lines[j].set_xdata(theta)
            self.vortex_lines[j].set_ydata(r)
            self.vortex_lines[j].set_marker(marker)
            self.vortex_lines[j].set_color(self.vortex_colours[v.circ])

            # And plot its trail - unless we just started animating
            if not i:
                continue
            
            # Get the trail in segments
            tl = self.trail_length
            trail_seg = v.get_trajectory(i, tl)
            
            # Set color and fade
            trail_color = np.array([0,0,0])
            tr_alpha = np.linspace(1, tl, tl)/tl
            
            trail_alpha = [np.append(trail_color, a) for a in tr_alpha]
            
            # Convert trail to segment and transfer data to plot
            self.trail_lines[j].set_segments(self.trail2segment(trail_seg))
            self.trail_lines[j].set_color(trail_alpha)
            
    """
    Converts a trajectory of type [[x0, y0], [x1, y1]], 
    to segments of type [ [[x0, y0], [x1, y1]], [[x1, y1], [x2, y2]]  ]
    which is what a LineCollection object expects in set_segment
    """
    def trail2segment(self, trail):
        segs = []
        # Start indexing at 1, since segments connect to last pt
        for i in np.arange(len(trail) - 1) + 1:
            s = [
                    np.flip(cart2pol([trail[i-1]])[0]),
                    np.flip(cart2pol([trail[i]])[0])
                ]
            segs.append(s)
        return segs

    def animate_trajectories(self):
        f = plt.figure()
        ax = f.add_subplot(111, polar = True)
        ax.grid(False)
        ax.set_xticklabels([])    # Remove radial labels
        ax.set_yticklabels([])    # Remove angular labels

        ax.set_ylim([0, self.settings['domain_radius']])    # And this turns out to be the radial coord. #consistency

        # Overconstructing axes for dipole/cluster lines, this can be optimized. At the very least, they need only be half size of vlines
        vlines = []
        dlines = []
        clines = []
                  
        for i, v in enumerate(self.vortices):
            # zorder value is arbitrary but high, we want vortices to be plotted on top of any other lines
            vlines.append(ax.plot([], [], '^', ls='', color = self.vortex_colours[v.circ], zorder = 1e3)[0])
            dlines.append(ax.plot([], [], color = self.dipole_colour)[0])
            clines.append(ax.plot([], [], color = self.cluster_colour)[0])

            lc = LineCollection([])
            ax.add_collection(lc)
            self.trail_lines.append(lc)

        self.vortex_lines = vlines
        self.dipole_lines = dlines
        self.cluster_lines = clines

        self.ani = animation.FuncAnimation(f, self.animate_trajectories_update, interval = 100)
        plt.show()
            
if __name__ == '__main__':
    fname = 'N10_T5_ATR0.01_90376.dat'  # Identifier
    
    pvm = PVM_Animation(fname)
    pvm.animate_trajectories()