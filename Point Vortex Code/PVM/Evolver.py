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
        initial_radius:    [Integer] Radial extent of initial vortex spawning
        spawn_sep:         [Float] Dipole spawning separation
        spawn_rate:        [Integer] Rate of Poisson process, relative to normalized max timesteps N = 1
        stirrer_rad:       [Float] Radius of stirring object
        stirrer_vel:       [Float] Velocity of stirring object
        
        ## Options for Monte-Carlo generation of states
        temperature:       [Float] Exactly what you think it is
        mc_skip:          [Integer] If specified, evolves system mc_sweep steps before saving a state
        mc_burn:           [Integer] If specified, evolves mc_burn steps before saving states
        mc_steps:          [Integer] Number of metropolis sweeps to do
        mc_enstrophy_tolerance: [Float] Tolerance for conservation of enstrophy
        mc_bounding_move_ratio: [Float] Divides the domain_radius by this to obtain bounding box for 
                                        where an mcmc step can move a vortex

"""

import numpy as np
import time
import pathlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.animation as animation
from itertools import chain, compress
from tqdm import tqdm
import collections
import pickle
from PVM.Utilities import pol2cart, cart2pol, eucl_dist, get_active_vortices

from PVM.Vortex import Vortex
from PVM.Conventions import Conventions

# For access to methods that calculate energy/spin
from PVM.Analysis import Analysis

class Evolver:

    def __init__(self,
                 n_vortices = 10,
                 coords = None,
                 circ = None,
                 T = 50,
                 dt = 0.01,
                 tol = 1e-8,
                 max_iter = 15,
                 annihilation_threshold = 1e-2,
                 verbose = True,
                 domain_radius = 3,
                 gamma = 0.01,
                 initial_radius = 3,
                 spawn_sep = .1,
                 spawn_rate = 0,
                 stirrer_rad = 1,
                 stirrer_vel = .3,
                 annihilate_on_boundary = True,
                 warm_file = None,
                 temperature = 0,
                 mc_skip = -1,
                 mc_burn = -1,
                 mc_steps = 100,
                 mc_enstrophy_tolerance = 1e-10,
                 mc_bounding_move_ratio = 10,
                 ):
        if warm_file:
            self.warm_start(warm_file)
            return
        
        self.annihilate_on_boundary = annihilate_on_boundary
        self.initial_radius = initial_radius
        self.domain_radius = domain_radius
        self.n_vortices = n_vortices
        self.max_n_vortices = n_vortices
        self.T = T

        self.mc_settings = {
                'temperature': temperature,
                'skip': mc_skip,
                'burn': mc_burn,
                'steps': mc_steps,
                'enstrophy_tol': mc_enstrophy_tolerance,
                'bbox': domain_radius/mc_bounding_move_ratio
                }
        
        self.dt = dt
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.gamma = gamma

        self.annihilation_threshold = annihilation_threshold
        self.circulations = []
        
        self.vortices = []

        if coords == None:
            self.init_random_pos()
        elif coords == "offcenter_2":
            assert n_vortices == 2
            self.initial_positions = np.array([[0.5*np.cos(np.pi/4), 0.5*np.cos(np.pi/4)],
                                       [0.5*np.cos(np.pi), 0.5*np.sin(np.pi)]])
            self.seed = -1
        elif coords == 'opposite_2':
            assert n_vortices == 2
            self.initial_positions = np.array([[.5, 0], [-.5, 0]])
            self.seed = -1
        else:
            self.initial_positions = coords
        if circ == None:
            self.init_circ()
        else:
            assert len(circ) == 1   # each frame has a corresp circulation array. Hence expect [ initial_circ ]
            self.circulations = circ

        self.trajectories = [ self.initial_positions ]
        
        # We want numpy array here for masking
        t0 = 0
        self.vortices = np.array([Vortex(p,c, t0) for p,c in zip(self.initial_positions, self.circulations[0][0, :])])
        
        self.spawn_times = []
        self.spawn_rate = spawn_rate
        self.spawn_sep = spawn_sep
        self.stirrer_rad = stirrer_rad
        self.stirrer_vel = stirrer_vel
        
    def warm_start(self, fname):
        fname = 'Datafiles/Evolution_' + fname
        with open(fname, "rb") as f:
            data = pickle.load(f)                
        
        self.set_trajectory_data(data)
        
        # ... and then evolve another batch forward

    # Generates vortex positions uniformly over unit disk
    def init_random_pos(self):
        # 5377
#        seed = 32257 # single annihilation at thr = 1e-3
#        seed = 44594 # double annihilation at thr = 1e-2
#        seed = 96329
        seed = np.random.randint(10**5)
        np.random.seed(seed)
        self.seed = seed
        print('seed: %d' % seed)

        r = np.sqrt(np.random.rand(self.n_vortices, 1))
        theta = 2*np.pi*np.random.rand(self.n_vortices, 1)

        # Second index is (x, y)
        self.initial_positions = np.hstack((pol2cart(r,theta)))

    # Generates the vorticity of each vortex
    def init_circ(self):
        c = np.zeros(self.n_vortices)
        h = len(c) // 2
        c[h:] = 1
        c[:h] = -1

        self.circulations = [np.flip(np.kron(np.ones((self.n_vortices, 1)), c))]
    
    # Spawns in vortex dipoles at a given rate, separation and position
    # Note that this function DOUBLES time per iteration(!). (check by setting spawn_rate = 0 or comment out call)
    # ^ depends on density/domain size/spawn rate of course. doubling at rate = 10, domain = 3
    def spawn(self, pos, t_frame):
        # Determine number of dipoles to spawn( for dt = 1e-2 and spawnrate ~ 1, this is mostly 1 or 0)
        ndp = np.random.poisson(self.spawn_rate*self.dt)
        
        for _ in np.arange(ndp):
            if self.verbose:
                tqdm.write("I be spawning %d vortex dipoles" % ndp)
            
            # Calculate spawn location
            omega = self.stirrer_vel/self.stirrer_rad
            
            # This may be too coarse.. rescale?
            n_orbits = 10 # class variable
            t_rs = t_frame*2*np.pi*n_orbits/int(self.T/self.dt)
            
            x, y = self.stirrer_rad*np.cos(omega*t_rs), self.stirrer_rad*np.sin(omega*t_rs)
            
            # we'd like to spawn vortices no closer than spawn_sep from any other vortex
            # recursively attempt to do this at different locations max_attempts times
            # if we hit max_attempts, we simply won't spawn
            max_attempts = 10
            ca = 0    # current attempt
            
            qp = np.array(pos) # so we can subtract of the position we have currently decided on
            while ca < max_attempts:
                ca = ca + 1
                # Calculate distance vectors from spawn loc to all other vortices
                dist_v = qp - np.array([[x,y]])
                # Calculate the corresponding norm
                dist_n = np.array([np.linalg.norm(v) for v in dist_v])
                
                # Are any closer than twice spawn sep? if so, update spawn loc and try again
                # (Note: twice spawn sep, because this is the cm coordinate and we spawn dipoles at +- spawn sep from it)
                if np.any(dist_n < 2*self.spawn_sep):
                    # We add gaussian noise with std equal to spawn sep and mean zero
                    dx, dy = np.random.normal(0, self.spawn_sep, 2)
                    x = x + dx
                    y = y + dy
                    continue
                
                # No neighbours spotted - spawn it
                break
            
            # Did we fail to find an isolated spot to spawn?
            if ca == max_attempts:  # Note that this misses a correct location found at the last attempt. Just increase max_attempts if that's a problem
                continue
            
            # Radial cm coordinates
            r, t = cart2pol(x, y)
            
            # Adjust negative and positve spin positions radially
            rn, rp = r + self.spawn_sep, r - self.spawn_sep
            
            # If we spawned outside boundary, try again
            if rn > self.domain_radius or rp > self.domain_radius:
                continue
            
            # Reconstruct cartesian coordinates as it is what we use in evolution 
            xp, yp = pol2cart(rp, t)
            xn, yn = pol2cart(rn, t)
            
            # Spawn is successful, so we update all relevant variables
            
            # Increase potential maximum, for animation reasons
            self.max_n_vortices = self.max_n_vortices + 2
            
            # And adjust current vortex number
            self.n_vortices = self.n_vortices + 2
            
            # Update vortex list
            vp = Vortex([xp, yp],  1, t_frame)
            vn = Vortex([xn, yn], -1, t_frame)
            self.vortices = np.append(self.vortices, [vp, vn])
            
            # Update trajectory
            pos = np.vstack((pos, [xp, yp]))
            pos = np.vstack((pos, [xn, yn]))
            
            # Update circulations. The -2 is here because we append column to old circulation first
            cp = np.ones(self.n_vortices - 2)
            
            # Add in new vorticity columns
            self.circulations[-1] = np.c_[self.circulations[-1], cp, -1*cp]
            
            # Add in another two (identical) rows
            crow = self.circulations[-1][0, :]
            self.circulations[-1] = np.r_[self.circulations[-1], [crow, crow]]
            
            
        return pos
        
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
        
        # TODO I have a sneaking suspicion zip(an_ind) would do exactly the same - no fancy footwork needed

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
        
        # Remove vortices that pass too close to the boundary(they annihilate with their image)
        # Note we have doubled the threshold here - otherwise they just start orbit
        rad_pos = np.array(cart2pol(pos))[:, 0]
        for ri, r in enumerate(rad_pos):
            if np.abs(self.domain_radius - r) < self.annihilation_threshold:
                an_ind = np.append(an_ind, ri)
        

        if len(an_ind) != 0:
            # Store which indices were annihilated at which time, lets us match up indices in the trajectories when animating
            amap = np.ones(self.n_vortices).astype(bool)
            amap[an_ind] = False
            
            # Update vortex count (used to dynamically generate images in rk4 etc)
            self.n_vortices = self.n_vortices - len(an_ind)

            # Delete vortices and their circulations. Slice at end because we delete across columns, which contain the distinct vortex circulations
            self.circulations.append(np.delete(self.circulations[-1], an_ind, axis = 1)[:-len(an_ind), :])
            
            # Get active vortices. Trajectories are constantly updated, so an_ind refers to active vortex list only
            living_mask = [v.is_alive(t_frame) for v in self.vortices]
            active_vortices = self.vortices[living_mask]
            
            # Get the vortices to kill
            dying_vortices = active_vortices[an_ind]
            
            # And annihilate them
            [dv.annihilate(t_frame) for dv in dying_vortices]
            
            # Finally, return updated trajectories
            return np.delete(pos, an_ind, axis = 0)
        
        # Memory inefficient. Devise scheme to only update vortex/circulation list on annihilation/spawning?
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
        
        # Conservative dynamics
        dx = -np.sum(circ.T*yy_temp.T/rr1, 0).T
        dy = np.sum(circ.T*xx_temp.T/rr1, 0).T 
        
        # Contribution from images
        dx = dx - np.sum(circ*yyp_temp.T/rr2, 1)
        dy = dy + np.sum(circ*xxp_temp.T/rr2, 1)
        
        # Dissipative dynamics
        dx = dx + self.gamma*np.sum(circ*circ.T*xx_temp.T/rr1, 0).T
        dy = dy - self.gamma*np.sum(circ*circ.T*yy_temp.T/rr1, 0).T

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

    def update_vortex_positions(self, cpos, c_time):
        living_mask = [v.is_alive(c_time) for v in self.vortices]
        av = self.vortices[living_mask]
        
        [v.set_pos(cp) for v, cp in zip(av, cpos)]


    """
    Integrates the full evolution of the system
    """
    def rk4(self, t_start = 0):
        ts = time.time()
        T = self.T
        dt = self.dt
        n_steps = int(T/dt)

#        c_time = t_start
        c_pos = self.initial_positions

        for i in tqdm(np.arange(n_steps-1) + 1):
            # Annihilation 
            # This function call also appends a new set of circulations to self.circulations - iff annihilation
            c_pos = self.annihilate(c_pos, i)
            
            #Spawn. As annihilate() adds a new circulation matrix, spawn() only modifies it
            c_pos = self.spawn(c_pos, i)

            # Adaptively evolve one timestep forward
            c_pos = self.rk4_error_tol(c_pos, i)
            self.update_vortex_positions(c_pos, i)

            # Add the corresp trajectory
            self.trajectories.append(c_pos)
#            c_time = t_start + i*dt

        tt = time.time() - ts
        if self.verbose:
            mins = tt // 60
            sec = tt % 60
            print('evolution complete after %d min %d sec' % (mins, sec))
        
        
        
    ########################
    ### MONTE-CARLO CODE ###
    ########################
    
    def metropolis(self):
#        self.mc_settings = {
#        'temperature': temperature,
#        'burn': mc_burn,
#        'steps': mc_steps,
#        'enstrophy_tol': mc_enstrophy_tolerance
#        }
        
        # Must be used with care, since there are no trajectories available etc.
        analysis = Analysis(None, self.get_trajectory_data())
        
        # Get stacked energy
        h_stacked = analysis.get_energy(0, True)
        
        # Equilibrate the system
        for i in np.arange(self.mc_settings['burn'] - 1) + 1:
            self.sweep(i, h_stacked, analysis) # not as bad as it looks, since obj-ref pass by value
            
        # Begin sweeping for statistics
        skip = self.mc_settings['skip']
        
        for j in (i+1) + np.arange(skip*self.mc_settings['steps']):
            if j % skip:
                # Save state
                pass
    
    """
    
    Performs one full sweep of the metropolis algorithm, i.e. attempts to move each vortex once
    
    We use two modifications: if vortex hits bdry, then we move it using specular reflection
    If this still fails to contain the vortex(as it might if it is moving parallelly close to bdry)
    then we do not move it.
    
    Secondly, the enstrophy Es is conserved. Thus we add an additional acceptance check |Es'-Es| < tol
    
    """
    def sweep(self, frame, h_stacked, analysis):
        beta = 1 / temperature
        es_tol = self.mc_settings['enstrophy_tol']
        
        # Stepping in order has a tendency to yield bad statistics. Not sure why...?
        v_tar = list(range(len(self.vortices)))
        np.random.shuffle(v_tar)
        
        # Do note that i and j are tuples
        for i in v_tar:
            # Grab target vortex
            v = self.vortices[i]
            
            # Energy of vortex prior to moving it
            energy_old = h_stacked[v.id]
            
            # Move vortex
            
            # Generate displacements from uniform distribution, center on zero and scale to boundingbox
            dr = self.mc_settings['bbox']*(np.random.rand(2) - 1/2)
            
            # Reflect if necessary
            if np.linalg.norm(v.get_pos() + dr) > self.domain_radius:
                new_pos = reflect(v.get_pos(), v.get_pos() + dr, self.domain_radius)
            
            # Energy of vortex after moving it 
            energy_new = analysis.get_energy(frame, False, v.id)
            
            # Energy change
            delta_E = energy_new - energy_old
            
            # Rejection sampling. Note that if delta_E <= 0, exp >= 1 and we always accept
            if np.random.uniform(0.0, 1.0) < np.exp(-beta * delta_E):
                self.spin_config[i] = (self.spin_config[i] + dtheta) % 2*np.pi
                self.energy = self.get_energy()
        
        
    
    def set_trajectory_data(self, data):
        settings = data['settings']
        
        for k, v in settings.items():
            setattr(self, k, v)
        
        self.trajectories = data['trajectories']
        self.circulations = data['circulations']
        self.vortices = data['vortices']
        
        
    
    def get_trajectory_data(self):
        settings = {
                'T': self.T,
                'dt': self.dt,
                'n_steps': int(self.T/self.dt),
                'domain_radius': self.domain_radius,
                'annihilation_threshold': self.annihilation_threshold,
                'tol': self.tol,
                'seed': self.seed,
                'max_n_vortices': self.max_n_vortices,
                'initial_radius': self.initial_radius,
                'spawn_sep': self.spawn_sep,
                'spawn_rate': self.spawn_rate,
                'stirrer_rad': self.stirrer_rad,
                'stirrer_vel': self.stirrer_vel,
                'gamma': self.gamma
                }    
        
        data = {
            'settings': settings,
            'trajectories': self.trajectories,
            'circulations': self.circulations,
            'vortices': self.vortices
                }  
        
        return data
            
    
    def save(self, fname = None):
        if fname == None:
            fname = Conventions.save_conventions(self.max_n_vortices, 
                                                      self.T, 
                                                      self.annihilation_threshold,
                                                      self.seed,
                                                      'Evolution')
            
        path = pathlib.Path(fname)
        if path.exists() and path.is_file():
            raise ValueError('This seed has already been evolved.')
            
        data = self.get_trajectory_data()
        with open(fname, "wb") as f:
            pickle.dump(data, f)


if __name__ == '__main__':
    pvm = Evolver()
    pvm.rk4()
    pvm.save()
