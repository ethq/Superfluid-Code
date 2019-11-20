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

from .Utilities import pol2cart, cart2pol, eucl_dist, get_active_vortices, get_active_vortex_cfg, get_vortex_by_id
from .Vortex import Vortex
from .PlotChoice import PlotChoice
from .Conventions import Conventions

class Animator:
    def __init__(self, 
                 fname = None,
                 evolution_data = None,
                 analysis_data = None,
                 animate_trails = True):
        
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
        
        # Other params
        self.animate_trails = animate_trails
        
        # Take relative to initial energy
        self.energies = self.energies - self.energies[0]
        
        # Remembering to close the files
        if fname:
            ef.close()
            af.close()
            
            
        self.n_steps = self.settings['n_steps']
            
        self.ani = None                    # For animation, handle required to stop 'on command'
        self.vortex_lines = []
        self.dipole_lines = []
        self.cluster_lines = []
        self.trail_lines = []
        
        # Various axes
        self.axes = {}
        
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
        
        # Support plotting/animating various data
        self.to_plot = PlotChoice.vortices_energy
        
    """
    Choose what to animate/plot. Possible values are enumerated in the PlotChoice class.
    """
    def choose_plot(self, choice):
        is_valid = choice in PlotChoice.get_possible_values()
        
        if not is_valid:
            raise ValueError('Invalid choice encountered in choose_plot(), PVM animation class')
        
        self.to_plot = choice

    """
    Could normalize to average energy per dipole, but probably have to do quite a few runs to get this statistic -
    which must also be obtained from some sort of stationary state
    """
    def update_energyPerVortex(self, i):
        # Avoiding some limit issues on the plot
        if i < 2 or i > len(self.trajectories)-1:
            return
        
        # Restrict ourselves to plotting the last n energies
        n_energies = 500
        ax = self.axes['energyPerVortex']
        
        start_i = np.max([i-n_energies, 0])
        
        times = np.linspace(start_i, i, i-start_i+1).astype(int)
        energies = self.energies[start_i:i+1]/len(self.circulations[i][0, :])
        
        ax.get_lines()[0].set_xdata(times)
        ax.get_lines()[0].set_ydata(energies)
        
        ax.set_xlim([start_i, i])
        ax.set_ylim([0, 1+np.max(energies)])
    
    def update_energies(self, i):
        # Avoiding some limit issues on the plot
        if i < 2 or i > len(self.trajectories)-1:
            return
        
        # Restrict ourselves to plotting the last n energies
        n_energies = 500
        ax = self.axes['energy']
        
        start_i = np.max([i-n_energies, 0])
        
        times = np.linspace(start_i, i, i-start_i+1).astype(int)
        energies = self.energies[start_i:i+1]
        
        ax.get_lines()[0].set_xdata(times)
        ax.get_lines()[0].set_ydata(energies)
        
        ax.set_xlim([start_i, i])
        ax.set_ylim([0, 1+np.max(energies)])
        
    def update_numberOfVortices(self, i):
        if i < 2 or i > len(self.trajectories)-1:
            return
        
        n_numbers = 500
        ax = self.axes['numberOfVortices']
        
        # Recall line order as they are constructed:
        # [total, free, dipoles, clusters]
        lines = ax.get_lines()
        
        # slightly sneaky way to get current number of active vortices
        total = len(self.circulations[i][0, :])
        dipoles = len(self.dipoles[i])
        clusters = len(self.clusters[i])
        free = total-dipoles-clusters
        
        data = [total, free, dipoles, clusters]
        
        start_i = np.max([i-n_numbers, 0])
        
        times = np.linspace(start_i, i, i-start_i+1).astype(int)
        
        for i in np.arange(4):
            lines[i].set_xdata(times)
            lines[i].set_ydata(data[i])
            
        ax.set_xlim([start_i, i])
        ax.set_ylim([0, 1+np.max(data)])
        
    
    """
    Since I am clearning all lines anyway, there is little point in keeping the y_lines as class members
    TODO: just store the axes then use get_lines() here and proceed as before
    """
    def update_trajectories(self, i):
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

            # And plot its trail - unless we just started animating or if we're not supposed to
            if not i or not self.animate_trails:
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


    def axsetup_vortices(self):
        ax = self.axes['vortices']
        
        # First add axis for the animation of vortices
#        ax = f.add_subplot(211, polar = True)
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
            
    def axsetup_energy(self):
        axe = self.axes['energy']
        
        axe.grid(False)
        axe.plot([], [])
        axe.set_title('Energy deviation')
        axe.set_xlabel('Frame')
        axe.set_ylabel('Deviation')
    
    """
    Sets up layout given a choice. Assumes the corresponding axes have been set up and are contained in self.axes
    """
    def layout(self, choice):
        # Gather settings with the same layout
        vortices_one_statistic = [PlotChoice.vortices_energy, 
                            PlotChoice.vortices_energyPerVortex,
                            PlotChoice.vortices_numberOfVortices]
        
        
        if choice in vortices_one_statistic:
            # get the relevant statistic axis
            if choice == PlotChoice.vortices_numberOfVortices:
                ax = self.axes['numberOfVortices']  
            elif choice == PlotChoice.vortices_energyPerVortex:
                ax = self.axes['energyPerVortex']
            elif choice == PlotChoice.vortices_energy:
                ax = self.axes['energy']
            
            # Adjust the statistic axis
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width, box.height*0.5])
        
            # Adjust the vortex axis
            ax = self.axes['vortices']
            box = ax.get_position()
            xratio = 1.8
            yratio = 1.8
            nbox = [box.x0-box.width*(xratio-1)/2, box.y0-box.height*(yratio-1)/2, box.width*xratio, box.height*yratio]
            
            # A little further down..
            nbox[-1] = nbox[-1] - 0.1
            ax.set_position(nbox)    
    
    def axsetup_energyPerVortex(self, f):
        axe = self.axes['energyPerVortex']
        
        axe.grid(False)
        axe.plot([], [])
        axe.set_title('Energy per vortex')
        axe.set_xlabel('Frame')
        axe.set_ylabel('Energy')
    
    def axsetup_numberOfVortices(self, f):
        axe = self.axes['numberOfVortices']
        
        axe.grid(False)
        
        # Create 4 line objects: all vortices, free vortices, dipoles and clusters
        axe.plot([], [], label = 'Total vortices')
        axe.plot([], [], label = 'Free vortices')
        axe.plot([], [], label = 'Dipoles')
        axe.plot([], [], label = 'Clusters')
        
        
        axe.set_title('Number statistics')
        axe.set_xlabel('Frame')
        axe.set_ylabel('Count')
        axe.legend()
    
    
    """
    Saves animation of the datafile passed in the constructor.
    choice:        [string] allowed values enumerated in PlotChoice class
    """
    def save_animation(self, choice):
        self.animate(choice, action = 'save')
    
    """
    Shows animation of the datafile passed in the constructor.f
    Wrapped by save_animation and show_animation
    
    choice:        [string] allowed values enumerated in PlotChoice class
    """
    def show_animation(self, choice):
        self.animate(choice, action = 'show')
    
    """
    Animates the datafile passed in the constructor. Wrapped by save_animation and show_animation
    
    choice:        [string] allowed values enumerated in PlotChoice class
    action:        [string] either 'show' or 'save. 
    """
    def animate(self, choice, action = 'save'):
        f = plt.figure()
        
        update_funcs = None
        
        if choice == PlotChoice.vortices_energy:
            self.axes['vortices'] = f.add_subplot(211, polar = True)
            self.axes['energy'] = f.add_subplot(212)
            
            self.axsetup_vortices()
            self.axsetup_energy()
            
            update_funcs = [self.update_energies,
                            self.update_trajectories,
                            ]
        
        elif choice == PlotChoice.vortices_numberOfVortices:
            self.axes['vortices'] = f.add_subplot(211, polar = True)
            self.axes['numberOfVortices'] = f.add_subplot(212)
            
            self.axsetup_vortices()
            self.axsetup_number()
            
            update_funcs = [self.update_trajectories,
                            self.update_numberOfVortices
                            ]
            
        elif choice == PlotChoice.vortices_energyPerVortex:
            self.axes['vortices'] = f.add_subplot(211, polar = True)
            self.axes['energyPerVortex'] = f.add_subplot(212)
            
            self.axsetup_vortices()
            self.axsetup_energypervortex()
            
            update_funcs = [self.update_trajectories,
                            self.update_energyPerVortex
                            ]
        else:
            raise ValueError('Unrecognized plotting choice in Animator')
            
        self.layout(choice)
        
        on_update = lambda i: [v(i) for v in update_funcs]
        self.ani = animation.FuncAnimation(f, on_update, interval = 1)
        
        # Save to file if desired
        if action == 'save':
            # Set up writer
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=15, metadata=dict(artist='Second Cheesegrater to the Left'), bitrate=1800)
            
            # Try to ensure we save entire animation
            self.ani.save_count = len(self.trajectories)
            
            if not self.fname:
                self.fname = Conventions.save_conventions(self.settings['max_n_vortices'], 
                                                      self.settings['total_time'],
                                                      self.settings['annihilation_threshold'],
                                                      self.settings['seed'],
                                                      data_type = 'Animation')
                
                
            
            print('saving animation')
            st = time.time()
            self.ani.save(self.fname, writer = writer)
            tt = time.time()-st
            mins = tt // 60
            sec = tt % 60
            print('saving finished after %d min %d sec' % (mins, sec))
        elif action == 'show':
            plt.show()
        else:
            raise ValueError('Uknown action passed to Animator.animate()')