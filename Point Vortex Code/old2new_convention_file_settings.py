# -*- coding: utf-8 -*-

import pickle
import numpy as np

"""
Created on Fri Nov 22 13:39:26 2019

@author: Zak
"""

"""
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

"""

"""
old convention -> new convention:
    
    total_time -> T
    timestep -> dt
    tolerance -> tol"
    
"""

fnames = ['N30_T100_S95642',
          'N668_T50_S68869',
          'N150_T50_S67603',
          'N30_T50_S92586'
          ]

def get_new_settings( settings ):
    data = settings.copy()
    to_change = {
            "total_time": "T",
            "timestep": "dt",
            "tolerance": "tol"
            }
    
    for k, v in data.items():
        if k in to_change.keys():
           data[to_change[k]] = v
           del data[k]

    return data

# Read file
for fname in fnames:
    fname_e = 'Datafiles/Evolution_' + fname + '.dat'
    with open(fname_e, "rb") as f:
        data = pickle.load(f)    
        
        new_settings = get_new_settings(data['settings'])
        data['settings'] = new_settings
        
    with open(fname, "wb") as f:
        pickle.dump(data, f)