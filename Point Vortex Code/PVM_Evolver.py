# -*- coding: utf-8 -*-
"""
Point Vortex Dynamics

Evolves a given initial configuration in time

"""


"""
    Initializes with:
        n_vortices:        [Integer] Number of vortices
        coords:            [Matrix or String or None] (N,2)-matrix of vortex positions,
                                              'opposite_2': two vortices at opposite ends
                                              'offcenter_2': two vortices with a cute angle between
                                              None: N vortices with random positions
        circ:              [Array]   Circulation signs of vortices
        T:                 [Integer] Total simulation time
        dt:                [Float]   Timestep
        tol:               [Float]   Tolerance for interval splitting


"""

import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.animation as animation
from itertools import chain
from tqdm import tqdm
import collections

class PVM_Evolver:

    def __init__(self,
                 n_vortices = 5,
                 coords = None,
                 circ = None,
                 T = 5,
                 dt = 0.01,
                 tol = 1e-8,
                 max_iter = 15,
                 annihilation_threshold = 1e-2,
                 verbose = True,
                 domain_radius = 5
                 ):
        self.domain_radius = domain_radius
        self.n_vortices = n_vortices
        self.max_n_vortices = n_vortices
        self.T = T
        self.dt = dt
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose

        self.trail_lines_max = 20
        self.annihilation_threshold = annihilation_threshold

        self.ani = None                    # For animation, handle required to stop 'on command'
        self.vortex_lines = []
        self.dipole_lines = []
        self.cluster_lines = []
        self.trail_lines = []

        self.trails = []
        self.circulations = []

        if coords == None:
            self.init_random_pos()
        elif coords == "offcenter_2":
            assert n_vortices == 2
            self.initial_positions = np.array([[0.5*np.cos(np.pi/4), 0.5*np.cos(np.pi/4)],
                                       [0.5*np.cos(np.pi), 0.5*np.sin(np.pi)]])
        elif coords == 'opposite_2':
            assert n_vortices == 2
            self.initial_positions = np.array([[.5, 0], [-.5, 0]])
        else:
            self.initial_positions = coords
        if circ == None:
            self.init_circ()
        else:
            assert len(circ) == 1   # each frame has a corresp circulation array. Hence expect [ initial_circ ]
            self.circulations = circ
        
        self.annihilation_map = collections.OrderedDict()

        self.trajectories = [ self.initial_positions ]
        self.dipoles = []
        self.clusters = []


    # Generates vortex positions uniformly over unit disk
    def init_random_pos(self):
        # 5377
#        seed = 32257 # single annihilation at thr = 1e-3
        seed = 44594 # double annihilation at thr = 1e-2
#        seed = np.random.randint(10**5)
        np.random.seed(seed)
        print('seed: %d' % seed)

        r = np.sqrt(np.random.rand(self.n_vortices, 1))
        theta = 2*np.pi*np.random.rand(self.n_vortices, 1)

        # Second index is (x, y)
        self.initial_positions = np.hstack((self.pol2cart(r,theta)))

    # Generates the vorticity of each vortex
    def init_circ(self):
        c = np.zeros(self.n_vortices)
        h = len(c) // 2
        c[h:] = 1
        c[:h] = -1

        self.circulations = [np.flip(np.kron(np.ones((self.n_vortices, 1)), c))]

    # Architecturally cleaner to annihilate before/after evolution, but does mean we compute the same thing twice.
    def annihilate(self, pos, t_frame):
        rr1 = self.calc_dist_mesh(pos)[4]

        # pick unique distances, e.g. set everything on and below diagonal to a lockout value(should exceed annihilation threshold)
        lockout = 1
        assert lockout > self.annihilation_threshold

        rr1 = np.triu(rr1, 1)
        rr1[rr1 == 0] = lockout


        # Get vortex indices to delete
        an_ind = np.where(rr1 < self.annihilation_threshold)
        # Combine element-wise. np.where returns in this case a 2d array with (row, col) indices, corresp vortex locations
        aind = np.empty((an_ind[0].shape[0])*2, dtype=int)
        aind[0::2] = an_ind[0]
        aind[1::2] = an_ind[1]
#        an_ind = list(chain.from_iterable(an_ind))
        an_ind = aind

        # Only annihilate opposite-signed vortices
        ai2 = np.array([])

        # Record where we are in the array. This is so we can check for uniqueness of vortex indices, so
        # that we do not attempt to annihilate the same vortex twice
        ai2c = 0
        for i, j in zip(an_ind[0::2], an_ind[1::2]):
            # A vortex may have multiple neighbours below threshold distance. If we have already annihilated it
            # then we skip this pair. To be precise we should annihilate only nearest neighbour, but this doesn't
            # seem computationally to be worth the effort - the possibility shrinks with threshold anyway.
            if np.any(np.isin(np.abs(ai2), [i,j])):
                continue

            # Store with circulation, so we do not annihilate equal-signed 'dipoles'
            s = self.circulations[-1][0, i]*self.circulations[-1][0, j]
            
            if s < 0:
                ai2 = np.append( ai2, s*np.array([i, j]) )
                ai2c = ai2c + 2

        ai2 = np.array(ai2)*-1
        assert (len(ai2) % 2) == 0
#        an_ind = (ai2[ai2 >= 0]).astype(int) # 'elegant' enough, but if vortex 0 gets deleted... shit hits the fan.
        an_ind = ai2.astype(int)

        if len(an_ind) != 0:
            # Store which indices were annihilated at which time, lets us match up indices in the trajectories when animating
            amap = np.ones(self.n_vortices).astype(bool)
            amap[an_ind] = False
            
            self.annihilation_map[t_frame] = amap
            
            # Update vortex count (used to dynamically generate images in rk4 etc)
            self.n_vortices = self.n_vortices - len(an_ind)

            # Delete vortices and their circulations. Slice at end because we delete across columns, which contain the distinct vortex circulations
            self.circulations.append(np.delete(self.circulations[-1], an_ind, axis = 1)[:-len(an_ind), :])
            return np.delete(pos, an_ind, axis = 0)

        # Highly memory inefficient. Also - time for a vortex class?
        self.circulations.append( self.circulations[-1] )
        return pos


    def calc_dist_mesh(self, pos):
        # unpack vectors of vortex positions
        xvec = pos[:, 0]
        yvec = pos[:, 1]
        # generate mesh of vortex positions
        xvec_mesh, yvec_mesh = np.meshgrid(xvec, yvec)

        # generate mesh of image positions
        xvec_im_mesh, yvec_im_mesh = np.meshgrid(self.domain_radius**2*xvec/(xvec**2 + yvec**2), self.domain_radius**2*yvec/(xvec**2 + yvec**2))

        # temp variables for calcing distance between voritces and between voritces & images
        yy_temp = yvec_mesh - yvec_mesh.T
        xx_temp = xvec_mesh.T - xvec_mesh

        yyp_temp = yvec_im_mesh - yvec_mesh.T
        xxp_temp = xvec_im_mesh.T - xvec_mesh

        # Switch to pdist maybe..
        # calc Euclidian distance between vortices (avoiding sigularity)
        rr1 = (xvec_mesh.T - xvec_mesh)**2 + (yvec_mesh.T - yvec_mesh)**2 + 1e-6*np.eye(self.n_vortices)

        # calc Euclidian distance between vortices and images
        rr2 = (xvec_mesh.T - xvec_im_mesh)**2 + (yvec_mesh - yvec_im_mesh.T)**2

        return xx_temp, yy_temp, xxp_temp, yyp_temp, rr1, rr2

    def evolve(self, t, pos):
        xx_temp, yy_temp, xxp_temp, yyp_temp, rr1, rr2 = self.calc_dist_mesh(pos)

        # calc vortex velocities
        circ = self.circulations[-1]

        dx = -np.sum(circ.T*yy_temp.T/rr1, 0).T - np.sum(circ*yyp_temp.T/rr2, 1)
        dy = np.sum(circ.T*xx_temp.T/rr1, 0).T + np.sum(circ*xxp_temp.T/rr2, 1)

        return 1/(2*np.pi)*np.hstack((dx[:, np.newaxis], dy[:, np.newaxis]))

    def rk4_step(self, pos, t, h):
        hh = 0.5*h
        h6 = h/6;

        k1 = self.evolve(t, pos)
        k2 = self.evolve(t+hh, pos + hh*k1)
        k3 = self.evolve(t+hh, pos + hh*k2)
        k4 = self.evolve(t+h, pos + h*k3)

        return pos + h6*(k1 + 2*k2 + 2*k3 + k4)

    def rk4_multi_step(self, pos, t_curr, j):
        nsteps = 2**j
        h = self.dt/nsteps

        cpos = pos
        c_time = t_curr
        t_start = t_curr

        for steps in np.arange(nsteps):
            cpos = self.rk4_step(cpos, c_time, h)
            c_time = t_start + steps*h
        return cpos

    def rk4_error_tol(self, pos, t_curr):
        # step at single splitting
        cpos = self.rk4_multi_step(pos, t_curr, 1)
        j = 2
        i = 0
        errorval = 1

        # step at increasing splits until convergence
        while errorval > self.tol and i < self.max_iter:
            npos = self.rk4_multi_step(pos, t_curr, j)
            errorval = np.sum(np.abs(cpos-npos))

            cpos = npos
            j = j + 1
            i = i + 1
        return cpos


    """
    Integrates the full evolution of the system
    """
    def rk4_fulltime(self, t_start = 0):
        ts = time.time()
        T = self.T
        dt = self.dt
        n_steps = int(T/dt)

        c_time = t_start
        c_pos = self.initial_positions

        for i in tqdm(np.arange(n_steps-1) + 1):
            # annihilation (passively updates self.circulations)
            c_pos = self.annihilate(c_pos, i)

            # Adaptively evolve one timestep forward
            c_pos = self.rk4_error_tol(c_pos, c_time)

            # Add the corresp trajectory
            self.trajectories.append(c_pos)
            c_time = t_start + i*dt


            # c_pos = self.update_spawning(c_pos)   ## todo: suppose we wish to compare to GPE sim, then we'd add spawning here

        tt = time.time() - ts
        if self.verbose:
            print('rk4_fulltime complete after %.2f s' % tt)
            
    """
    Converts cartesian to polar coordinates. Returns [r, theta].
    If y is not supplied, array MUST BE of form
    x = [
            [x0, y0],
            [x1, y1],
            ...
        ]
    """
    def cart2pol(self, x, y = None):
        # If y is not supplied, we provide some other behaviour
        if np.any(y == None):
            out = []
            for p in x:
                out.append( list(self.cart2pol_scalar(p[0], p[1])) )
            return out
        else:
            return self.cart2pol_scalar(x,y)
    
    def cart2pol_scalar(self, x, y):
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return(rho, phi)

    def pol2cart(self, rho, phi):
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return(x, y)

#    def save_trajectories(self, fname = None):
#        if fname == None:
#            fname = "VortexTrajectories_N%d_T%d" % (self.max_n_vortices, self.T)
#        np.savez(fname, 
#                 trajectories = self.trajectories, 
#                 circulations = self.circulations, 
#                 dipoles = self.dipoles,
#                 clusters = self.clusters,
#                 annihilation_map = self.annihilation_map)
#
#    def load_trajectories(self, fname):
#        files = np.load(fname)
#        self.circulations = files['circulations']
#        self.trajectories = files['trajectories']
#        self.dipoles = files['dipoles']
#        self.clusters = files['clusters']
#        self.annihilation_map = files['annihilation_map']

    """
       All cluster code expects a vortex configuration (cfg) in the form of an (N,2) matrix of positions

       Could be considerably beautified by introducing a vortex class; labeling them by a unique id removes messy index crutch
    """


if __name__ == '__main__':
    pvm = PVM_Evolver()
    pvm.rk4_fulltime()