# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 14:09:59 2020

@author: Zak
"""

import numpy as np
from PVM.Utilities import pol2cart, cart2pol

class CONFIG_STRAT:
    UNIFORM = 'uniform'
    DOUBLE_CLUSTER = 'double_cluster'
    OFFCENTER_2 = 'offcenter_2'
    OPPOSITE_2 = 'opposite_2'
    SINGLE_CLUSTER = 'single_cluster'
    SINGLE_VORTEX_IN_CLOUD = 'single_vortex_in_cloud'
    
    CIRCS_ALL_POSITIVE = 'circs_all_positive'
    CIRCS_ALL_BUT_ONE_POSITIVE = 'circs_all_but_one_positive'
    CIRCS_EVEN = 'circs_even'

"""

Class to generate various more or less annoying initial conditions (position, vorticity)

"""
class Configuration:
    """
    n:     [Integer] number of vortices
    r:     [Integer] radius of domain
    strategy: [String] as enumerated above
    max_attempts: [Integer] will try to generate a configuration this many times. default failure condition:
                            vortex outside domain.
    params:     [Dictionary] extra parameters if needed for a certain strategy
    validation_options:  [Dictionary] 
    """
    def __init__(self,
                 n,
                 r,
                 coord_strategy,
                 circ_strategy,
                 seed = None,
                 params = {},
                 validation_options = {},
                 max_attempts = 1000):
        
        self.n_vortices = n
        self.domain_radius = r
        self.pos = [[0,0]]
        self.circ = [[0,0]]
        
        # Set up coordinates first. These can fail, as opposed to circulations
        ctr = 0
        is_valid = False
        while ctr < max_attempts:
            if not seed:
                seed_ = np.random.randint(10**9)
            else:
                seed_ = seed
            np.random.seed(seed_)
                      
            
            getattr(self, coord_strategy)(params)            
            is_valid = self.validate(validation_options)
            
            if is_valid:
                break
            
            ctr = ctr + 1
        
        if not is_valid:
            raise ValueError("Unable to generate requested configuration")
        
        self.seed = seed_  
        print(f'Configuration seeded with: {self.seed}.')
        
        # Set up circulations
        getattr(self, circ_strategy)(params)
        self.format_circulation()
        
      
    """
    p:     [Dictionary] Must contain keys:
                        'cartesian': if true, generates uniform cartesian coords (which are not uniform in polar)
                        
    """
    def uniform(self, p):
        cartesian = p['cartesian']
        if cartesian:
            x = (np.random.rand(self.n_vortices, 1))*self.domain_radius
            y = (np.random.rand(self.n_vortices, 1))*self.domain_radius
            
            self.pos = np.hstack((x,y))
            return
        
        r = np.sqrt(self.domain_radius**2*np.random.rand(self.n_vortices, 1))
        theta = 2*np.pi*np.random.rand(self.n_vortices, 1)

        # Second index is (x, y)
        self.pos = np.hstack((pol2cart(r,theta)))
    
    
    """
    p:     [Dictionary] Must contain keys:
                        'center_ratio': center of cluster at +- domain_radius*center_ratio
                        'sigma_ratio': standard deviation of generated clusters
    """
    def double_cluster(self, p):
        center_ratio = p['center_ratio']
        sigma_ratio = p['sigma_ratio']
        
        # For simplicity just demand an even number of vortices
        assert self.n_vortices % 2 == 0
        
        # Spontaneously break symmetry ;) by choosing antipodal centres
        c1x = np.ones(self.n_vortices // 2)*self.domain_radius*center_ratio
        c2x = -c1x
        
        # Join to add normal distributed radial displacements
        c = np.hstack((c1x, c2x))[:, np.newaxis]
        
        # Std dev for radial displacements around centre. 
        sigma = np.sqrt(sigma_ratio*self.domain_radius)
        
        # Generate vortex positions around each vortex. By construction the first half have positive circ
        r = sigma*np.random.randn(self.n_vortices, 1)
        theta = 2*np.pi*np.random.rand(self.n_vortices, 1)
        
        dx, dy = pol2cart(r, theta)
        
        self.pos = np.hstack((c+dx, dy))
    
    def offcenter_2(self, p):
        self.pos = np.array([[0.5*np.cos(np.pi/4), 0.5*np.cos(np.pi/4)],
                                       [0.5*np.cos(np.pi), 0.5*np.sin(np.pi)]])
    
    def opposite_2(self, p):
        self.pos = np.array([[.5, 0], [-.5, 0]])
    
    
    """
    p:     [Dictionary] Must contain keys:
                        'center': [Tuple/List] in cartesian coordinates, e.g. (x,y)
                        'sigma': standard deviation of cluster
    """
    def single_cluster(self, p):
        # Set defaults
        if not p:
            p['center'] = [0, 0]
            p['sigma'] = .1*self.domain_radius
        
        # Define a cluster center
        mu = np.array(p['center'])
        
        # Take vortices normally distributed around center
        sigma = p['sigma']
        pos = sigma*np.random.randn(self.n_vortices, 2)
        pos[:, 0] = pos[:, 0] + mu[0]
        pos[:, 1] = pos[:, 1] + mu[1]
        
        self.pos = pos
    
    
    """
    p:     [Dictionary] Expects keys:
                        'center': [Tuple/List] in cartesian coordinates, e.g. (x,y)
                        'sigma': standard deviation of cluster
    """
    def single_vortex_in_cloud(self, p):
        # Set defaults if no parameters passed
        if not p:
            p['center'] = np.array([1e-4, 1e-4])
            p['sigma'] = .1*self.domain_radius
        
        self.single_cluster(p)
        
        mu = np.array(p['center'])
        
        # Center particle zero
        self.pos[0, :] = mu
        
    
    def circs_all_positive(self, p):
        self.circ = np.ones(self.n_vortices)
    def circs_all_negative(self, p):
        self.circ = np.ones(self.n_vortices)*-1
    
    def circs_all_but_one_positive(self, p):
        c = np.ones(self.n_vortices)
        
        # We later apply an np.flip(), hence this gives the vortex with id = 0 negative circ
        c[-1] = -1
        self.circ = c
    
    
    def circs_even(self, p):
        c = np.ones(self.n_vortices)
        h = self.n_vortices // 2
        c[:h] = -1
        
        self.circ = c
    
    
    """
    Validates the generated positions. Options may be supplied to validate on additional conditions.
    
    Supported options:
        minimum_separation:     [Float] Vortices cannot be generated closer than the given value.
        
    """
    def validate(self, options = {}):        
        # Grab polar coordinates fist
        pos = np.array(cart2pol(self.pos))
        r = pos[:, 0]
        
        # Is any position outside the boundary?
        mask = np.sum(np.array([rad > self.domain_radius for rad in r]).astype(int))
        
        # If so, the sum of mask will be greater than zero, and we fail
        if mask > 0:
            return False
        
        # If no options are supplied we are done (this is here to prevent key errors)
        if not options:
            return True
        
        # Validate on minimum separation
        if options['minimum_separation']:
            minsep = options['minimum_separation']
            
            # Grab vortex cartesian coordinates
            pos = self.pos
            
            # Loop over every vortex and check its distance to neighbours. Could be optimized by deleting a vortex after checking.
            for i, p in enumerate(pos):
                d = np.linalg.norm(pos - p, axis = 1)
                
                # No self-contribution
                d = np.delete(d, i)
                
                # Any vortex closer than minsep? If so, fail immediately
                if (d < minsep).any():
                    return False
        
        # We got this far - must be all good!
        return True

    """
    Since we have vectorized evolution, we'll need to put the circulations into a matrix
    Thus this must always be called after a particular set of circulations have been
    generated.
    """
    def format_circulation(self):
        self.circulations = [np.flip(np.kron(np.ones((self.n_vortices, 1)), self.circ))] 
       
        
#            seed = 32257 # single annihilation at thr = 1e-3
#            seed = 44594 # double annihilation at thr = 1e-2