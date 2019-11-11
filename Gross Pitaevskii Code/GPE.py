# -*- coding: utf-8 -*-
"""
GPE 
"""

import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
from Stirrer import Stirrer
from matplotlib import cm

"""
    Class for simulating the GPE in various potentials
"""
class GPE():
    def __init__(self,
                 potential = 'harmonic',
                 stirrer = None,
                 fourier_modes = 2**9,
                 real_space_size = 2**8,
                 total_time = 5000,
                 timestep = 1e-1,
                 interaction_strength = 1,
                 dissipation = 0.01,
                 batch_size = 1,
                 imaginary_time = False,
                 initial_state = 'tf'
                 ):
        
        self.potential = potential
        
        self.batch_size = batch_size
                
        self.N = fourier_modes          # Fourier modes
        self.L = real_space_size        # Real space domain size
        self.h = self.L/self.N          # Grid size
        
        self.T = total_time             # Deprecated
        self.dt = timestep
        self.TN = np.round(self.T/self.dt) # Deprecated
        
        self.beta = interaction_strength         # Interaction strength(positive = repulsive)
        self.gamma = dissipation                 # Dissipative strength
        
        
        self.RTF = 0.4*self.N           # Desired radius, _should_ be propagated in imtime to reach gs, stored.
        
        self.construct_grid()
        self.construct_hamiltonian()
        
        if stirrer == None:
            stirrer = {
                         'type': 'gaussian',
                         'params': 
                     {'radius': .3*self.RTF,
                      'width': 4,
                      'velocity': .5, 
                      'strength': 1.4
                     }
                     }
        
        self.stirrer = Stirrer(stirrer['type'], stirrer['params'], (self.X, self.Y)) # Memory wasteful to store a double grid, but..
        
        # Initial state, taken to be the thomas-fermi state. add option to load in future
        self.Vp = self.V(0)
        u0 = 1 - self.Vp
        u0[u0 < 0] = 0
        self.u0 = np.sqrt(u0)
        self.uc = u0
        
        # Maybe only do this in the evolution
        if (imaginary_time):
            self.dt = self.dt*1j
        
        # iteration counter
        self.i = 0

    def construct_grid(self):
        # Create momentum space grid
        halfpoint = np.round(self.N/2).astype(int)
        k = 2*np.pi/self.L*(np.concatenate((
                np.linspace(0, halfpoint, halfpoint+1),
                np.linspace(-(halfpoint-1), -1, halfpoint-1)
                )))
        
        [KX, KY] = np.meshgrid(k, k)    

        # Create real space grid
        x = np.linspace(-halfpoint, halfpoint-1, 2*halfpoint)*self.h
        y = np.linspace(-halfpoint, halfpoint-1, 2*halfpoint)*self.h
        [X, Y] = np.meshgrid(x,y)           
        
        # And store
        self.X, self.Y = X, Y
        self.KX, self.KY = KX, KY
        
    def construct_hamiltonian(self):
        self.lap = (1j + self.gamma)*(1-0.5*(self.KX**2 + self.KY**2))   # Laplacian + chempot term(presumably? check)
        self.elap = np.exp(self.lap*self.dt)
        
        self.Vtrap = 0
        if (self.potential == 'harmonic'):
            g = np.sqrt(2)/self.RTF                                 # Trap potential frequency
            self.Vtrap = 0.5*g**2*(self.X**2  + self.Y**2)          # (Fixed) Trap potential
    
    # Evolves the wavefunction batch_size timesteps forward
    def update(self):
        for j in np.arange(self.batch_size):
            self.step()

    # Gives the potential at time t
    def V(self, t):
        return self.Vtrap + self.stirrer.V(t)
    
    
    def step(self):
        V0 = self.Vp
        uc = self.uc
        
        V1 = self.V(self.i*self.dt)
        uf = fft.fftn(uc)
        pf = 1j + self.gamma
        
        N0 = -pf*fft.fftn((V0 + uc*np.conj(uc))*uc)
        pred = self.elap*uf + N0*(self.elap - 1)/self.lap
        predr = fft.ifftn(pred)
        
        N1 = (-pf*fft.fftn((V1 + np.conj(predr)*predr)*predr) - N0)/self.dt
        uc = fft.ifftn(pred + N1*(self.elap - (1+self.lap*self.dt))/self.lap**2)

        self.uc = uc
        self.Vp = V1
        
        self.i = self.i + 1



#%ucv = ucv/norm(ucv);

# Regular timesplitting doesn't work for some reason.. 

#fig = plt.figure()
#ax = fig.gca(projection='3d')
#ax.view_init(azim=0, elev=90)
#surf = ax.plot_surface(X, Y, np.conj(uc)*np.conj(uc), cmap = cm.jet)
#s = surf(X, Y, conj(uc).*uc);
#set(s, 'LineStyle', 'none');
#view(0,90)
#axis tight

#V0 = Vs(0)     # Potential at previous timestep
#for i in np.arange(TN-1)+1:
#    V1 = Vs(i*dt)
#    uf = fft.fftn(uc)
#    
#    N0 = -pf*fft.fftn((V0 + uc*np.conj(uc))*uc)
#    pred = elap*uf + N0*(elap - 1)/lap
#    predr = fft.ifftn(pred)
#    
#    N1 = (-pf*fft.fftn((V1 + np.conj(predr)*predr)*predr) - N0)/dt
#    uc = fft.ifftn(pred + N1*(elap - (1+lap*dt))/lap**2)
#    
#    V0 = V1



