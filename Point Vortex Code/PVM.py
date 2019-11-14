# -*- coding: utf-8 -*-
"""
Point Vortex Dynamics

Unit disk for now(remember this enters in calculation of image vortex positions)

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

class PVM:

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
        for j,v in enumerate(cfg):
            if j == i:
                # don't calculate self-distance. needless optimization
                assert(cfg[i, :] == cfg[j, :])
                
            d = self.eucl_dist(cv, v)
            
            if d > r and d < r + dr:
                ann = ann + 1
        return ann
    
    def energy(self, pos):
        pass


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

    def animate_trajectories_update(self, i):
        t = self.trajectories

        if i >= len(self.trajectories):
            self.ani.event_source.stop()
            return

        cfg = self.trajectories[i]
        dipoles = self.dipoles[i]
        clusters = self.clusters[i]

        # Plot dipoles first
        for dl in self.dipole_lines:
            dl.set_xdata([])
            dl.set_ydata([])

        d_counter = 0
        for k,j in zip(dipoles[0::2], dipoles[1::2]):
            x = np.array([cfg[k, 0], cfg[j, 0]])[:, np.newaxis]
            y = np.array([cfg[k, 1], cfg[j, 1]])[:, np.newaxis]

            r, t = self.cart2pol(x, y)
            self.dipole_lines[d_counter].set_xdata(t)
            self.dipole_lines[d_counter].set_ydata(r)
            d_counter = d_counter + 1


        #T hen plot clusters
        c_counter = 0

        for cl in self.cluster_lines:
            cl.set_xdata([])
            cl.set_ydata([])

        for c in clusters:
            x, y = cfg[c].T
            r, t = self.cart2pol(x, y)
            self.cluster_lines[c_counter].set_xdata(t)
            self.cluster_lines[c_counter].set_ydata(r)

            c_counter = c_counter + 1

        cluster_ids = list(chain.from_iterable(clusters))

        # Clear out trail lines
        for tr in self.trail_lines:
            tr.set_segments([])


        # Plot vortices themselves
        if i >= 1:
            # In case of annihilation, remove corresponding lines
            pvort = len(self.trajectories[i-1])
            cvort = len(self.trajectories[i])
            
            if cvort < pvort:
                if self.verbose:
                    print('Annihilation of %d vortices occured.' % (pvort-cvort))
                
                
                for vl in self.vortex_lines[:-(pvort-cvort)]:
                    vl.set_xdata([])
                    vl.set_ydata([])

        colors = ['#383535', '#bd2b2b']
        for j in np.arange(len(cfg[:,0])):
            x, y = cfg[j, :]
            r, theta = self.cart2pol(x, y)

            ci = int((self.circulations[i][0, j]+1)/2)

            # Plot vortices themselves
            marker = '^'

            if j in dipoles:
                marker = 'o'
            elif j in cluster_ids:
                marker = 's'

            self.vortex_lines[j].set_xdata(theta)
            self.vortex_lines[j].set_ydata(r)
            self.vortex_lines[j].set_marker(marker)
            self.vortex_lines[j].set_color(colors[ci])

            # And plot its trail - unless we just started animating
            if not i:
                continue

            # Get the trail in segments
            trail_seg = self.get_trail_fp(j, i, 10)
            ltrs = len(trail_seg)
#            trail_color = np.array([189, 214, 79])/255
            trail_color = np.array([0,0,0])
            tr_alpha = np.linspace(1, ltrs, ltrs)/ltrs
            
            trail_alpha = [np.append(trail_color, a) for a in tr_alpha]
            self.trail_lines[j].set_segments(trail_seg)
            
            self.trail_lines[j].set_color(trail_alpha)

        lines = self.vortex_lines
        lines.extend(self.dipole_lines)
        lines.extend(self.cluster_lines)
#        return lines


    def animate_trajectories(self):
        f = plt.figure()
        ax = f.add_subplot(111, polar = True)
        ax.grid(False)
        ax.set_xticklabels([])    # Remove radial labels
        ax.set_yticklabels([])    # Remove angular labels

        ax.set_ylim([0, self.domain_radius])    # And this turns out to be the radial coord. #consistency

        # Overconstructing axes for dipole/cluster lines, this can be optimized. At the very least, they need only be half size of vlines
        vlines = []
        dlines = []
        clines = []
        colors = ['black', '#bd2b2b']
                  
        for i in np.arange(self.max_n_vortices):
            ci = int((self.circulations[0][0, i]+1)/2)
            
            # zorder value is arbitrary but high, we want vortices to be plotted on top of any other lines
            vlines.append(ax.plot([], [], '^', ls='', color = colors[ci], zorder = 1e3)[0])
            dlines.append(ax.plot([], [], color = '#c0e39d')[0])
            clines.append(ax.plot([], [], color = '#57769c')[0])

            lc = LineCollection([])
            ax.add_collection(lc)
            self.trail_lines.append(lc)

        self.vortex_lines = vlines
        self.dipole_lines = dlines
        self.cluster_lines = clines

        self.ani = animation.FuncAnimation(f, self.animate_trajectories_update, interval = 100)
        plt.show()

    def save_trajectories(self, fname = None):
        if fname == None:
            fname = "VortexTrajectories_N%d_T%d" % (self.max_n_vortices, self.T)
        np.savez(fname, 
                 trajectories = self.trajectories, 
                 circulations = self.circulations, 
                 dipoles = self.dipoles,
                 clusters = self.clusters,
                 annihilation_map = self.annihilation_map)

    def load_trajectories(self, fname):
        files = np.load(fname)
        self.circulations = files['circulations']
        self.trajectories = files['trajectories']
        self.dipoles = files['dipoles']
        self.clusters = files['clusters']
        self.annihilation_map = files['annihilation_map']

    """
       All cluster code expects a vortex configuration (cfg) in the form of an (N,2) matrix of positions

       Could be considerably beautified by introducing a vortex class; labeling them by a unique id removes messy index crutch
    """
    def cluster_analysis(self, cfg):
        dipoles = self.find_dipoles(cfg)

        clusters = self.find_clusters(cfg, dipoles)


        return dipoles, clusters

    def find_clusters(self, cfg, dipoles):
#        circs = self.circulations[0, :]
        circs = cfg[1][0, :]
        clusters = []
        cluster_ids = []

        for i in np.arange(len(cfg[0][:, 0])):
            if i in dipoles or i in cluster_ids:
                continue

            # Find cluster partners
            tar = cfg[0][i, :]
            ts = circs[i]

            _, tar_enemy_dist = self.find_nn(cfg, i, True)

            cluster = []

            for j in np.arange(len(cfg[0][:, 0])):
                # Skip if dipole or of opposite sign
                if j in dipoles or circs[j] != ts or j==i:
                    continue

                # We found a same-sign not-dipole vortex.
                # Check their distance and their distance to nearest opposite sign

                dist_friend = self.eucl_dist(tar, cfg[0][j, :])

                if (tar_enemy_dist < dist_friend):
                    continue

                _, dist_enemy = self.find_nn(cfg, j, True)

                if (dist_enemy < dist_friend):
                    continue

                # Friends are closer than either are to an enemy, cluster them
                if not len(cluster):
                    cluster.append(i)

                cluster.append(j)

            cluster_ids.extend(cluster)
            if len(cluster):
                clusters.append(cluster)

        return clusters

    """
        Rule: if two vorticies are mutually nearest neighbours, we classify them as a dipole
    """
    def find_dipoles(self, cfg):
        dipoles = []
        circs = cfg[1][0, :]

#        circs = self.circulations[0, :]

        # Loop over all vortices
        for i in np.arange(len(cfg[0][:, 0])):
            # If already classified as dipole, skip
            if i in dipoles:
                continue
            # Find nearest neighbour of this vortex
            j,_ = self.find_nn(cfg, i)

            # ... and nn of nn
            i2,_ = self.find_nn(cfg, j)

            # Mutual nearest neighbour found, classify as dipole if signs match
            if ((i2 == i) and (circs[i] != circs[j])):
                dipoles.append(i)
                dipoles.append(j)

        return dipoles

    """
    find nearest neighbour of vortex number i. if opposite is true, find only neighbour of opposite circulation
    """
    def find_nn(self, cfg, i, opposite = False):
        tar = cfg[0][i, :]
        smallest_dist = np.Inf
        nn = -1

#        circs = self.circulations[0, :]
        circs = cfg[1][0, :]

        for j in np.arange(len(cfg[0][:, 0])):
            if j == i or (opposite and (circs[i] == circs[j])):
                continue

            dist = self.eucl_dist(tar, cfg[0][j, :])
            if (dist < smallest_dist):
                smallest_dist = dist
                nn = j

        return nn, smallest_dist


    def eucl_dist(self, v1, v2):
        d = v1-v2
        return d @ d.T


    """
    Input:
        i            [Integer] indexes a configuration in trajectories. Trails computed for j < i
        fixed_len    [Integer] if given, locks all trails to fixed length in real space
        fixed_pts    [Integer] if given, locks all trails to a fixed number of points, so faster vortices have longer trails

    """
    def compute_trails(self, vid, tid, fixed_len = -1, fixed_pts = 20):
        # return trajectory points('trails') from times prior to tid, hence the -1
        tid = tid - 1

        assert fixed_len > 0 or fixed_pts > 0
        assert tid < len(self.trajectories)

        if fixed_pts:
            self.compute_trails_fp(tid, fixed_pts)
        elif fixed_len:
            self.compute_trails_fl(tid, fixed_len)

    def get_trail_fp(self, vid, tid, fp):
        # np.max expects an iterable and returns largest value therein. second argument is axis to calc along
        si = np.max([tid-fp, 0])

        # no trail to return!
        if si == tid:
            return None

        trail = self.trajectories[si:tid]
        shortest_trail_len = len(trail[-1])
        n_trail = []
        
        # Reconstruct trails taking annihilations into account, e.g. index matching
        for i, tr in enumerate(trail):            
            # Annihilation has occured
            if len(trail[i]) > shortest_trail_len:
                # The vortex index 'vid' refers to the trajectory at t = tid, or trails[-1]. 
                # Hence we must "annihilate" down previous maps to get the correct trajectories
                
                n_trail.append( self.annihilate_traj_to(trail[i], i+si, tid) )
            else:
                n_trail.append( trail[i] )
        
        trail = [tr[vid, :] for tr in n_trail]
        
        return self.trail2segment(trail)
    
    def annihilate_traj_to(self, traj, start_t, stop_t):
        tc = traj
        
        # Successively apply annihilation maps from start_t to stop_t
        for t, mask in self.annihilation_map.items():
            if t <= start_t:
                continue
            elif t >= stop_t:
                break
            
            tc = tc[mask]
        return tc
    
    def trail2segment(self, trail):
        segs = []
        # Start indexing at 1, since segments connect to last pt
        for i in np.arange(len(trail) - 1) + 1:
            s = [
                    np.flip(self.cart2pol([trail[i-1]])[0]),
                    np.flip(self.cart2pol([trail[i]])[0])
                ]
            segs.append(s)
        return segs
            

    def compute_trails_fl(self, i, fl):
        pass

    def compute_trajectory_clusters(self):
        dipoles = []
        clusters = []

        for i in np.arange(len(self.trajectories)):
            d,c = self.cluster_analysis([self.trajectories[i], self.circulations[i]])
            dipoles.append(d)
            clusters.append(c)

        self.dipoles = dipoles
        self.clusters = clusters

    def plot_configuration(self, cfg, dipoles = [], clusters = []):
        f = plt.figure()
        ax = f.add_subplot(111, polar = True)
        ax.grid(False)
        ax.set_xticklabels([])    # Remove radial labels
        ax.set_yticklabels([])    # Remove angular labels

        ax.set_ylim([0, 1])    # And this turns out to be the radial coord. #consistency


        # Plot dipoles first
        if len(dipoles):
            for i,j in zip(dipoles[0::2], dipoles[1::2]):
                x = np.array([cfg[i, 0], cfg[j, 0]])[:, np.newaxis]
                y = np.array([cfg[i, 1], cfg[j, 1]])[:, np.newaxis]

                r, t = self.cart2pol(x, y)
                ax.plot(t, r, '-', color = 'green')

        # Then plot clusters
        for c in clusters:
            x, y = cfg[c].T
            r, t = self.cart2pol(x, y)

            ax.plot(t, r, '-', color = 'blue')

        cluster_ids = np.array(clusters).flatten()

        # Plot vortices themselves
        colors = ['black', 'red']
        for i in np.arange(len(cfg[:,0])):
            x = cfg[i, 0]
            y = cfg[i, 1]


            r, theta = self.cart2pol(x, y)

            ci = int((self.circulations[0, i]+1)/2)

            marker = '^'

            if i in dipoles:
                marker = 'o'
            elif i in cluster_ids:
                marker = 's'


            ax.plot(theta, r, marker, color = colors[ci])



        plt.show()

    def plot_trajectories(self):
        traj = self.trajectories

        f = plt.figure()
        ax = f.add_subplot(111, polar = True)
        ax.grid(False)
        ax.set_xticklabels([])    # Remove radial labels
        ax.set_yticklabels([])    # Remove angular labels

        ax.set_ylim([0, 1])    # And this turns out to be the radial coord. #consistency
#        plt.xkcd()

        for i in np.arange(self.n_vortices):
            xy = traj[:, i, :]
            x = xy[:, 0]
            y = xy[:, 1]


            r,theta = self.cart2pol(x,y)

            ax.plot(theta, r)

        plt.show()

if __name__ == '__main__':
    pvm = PVM()

#    pvm.load_trajectories('VortexTrajectories_N10_T5.npz')

#    c = pvm.trajectories[100, :, :]
#    pvm.compute_trails(100)


#    pvm.load_trajectories('VortexTrajectories_N30_T5.npz')
#    pvm.animate_trajectories()

#    c = pvm.trajectories[30, :, :]
#    ds, cs = pvm.cluster_analysis(c)
#    pvm.plot_configuration(c, ds, cs)

    pvm.rk4_fulltime()
    pvm.compute_trajectory_clusters()
    pvm.animate_trajectories()
#    pvm.plot_trajectories()
#    pvm.save_trajectories()