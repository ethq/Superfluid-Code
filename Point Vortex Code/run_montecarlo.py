# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 13:02:46 2020

@author: zakla
"""

import ctypes
import PVM as pvm
import numpy as np

# First set up initial conditions
n_vortices = 50
domain_radius = 2000
annihilate_at_radius = 1988
T = 5015

params = {
        'center': [1e-4, 1e-4],
        'sigma': 40 ## Previous value: 40
        }

cfg = pvm.Configuration(
        n_vortices,
        domain_radius,
        pvm.CONFIG_STRAT.UNIFORM,
        pvm.CONFIG_STRAT.CIRCS_EVEN,
        seed = None,
        params = None,
        validation_options = 
        {
                'minimum_separation': 1e-2
        }
        )
ev_config = {
    'n_vortices': n_vortices,
    'dt': .1,
    'domain_radius': domain_radius,
    'annihilate_at_radius': annihilate_at_radius,
    'gamma': .3,
    'T': T,
    'spawn_rate': 0,
    'cfg': cfg
    # 'tf_profile': tf_profile
    }

evolver = pvm.Evolver(**ev_config)


ctypes.windll.user32.FlashWindow(ctypes.windll.kernel32.GetConsoleWindow(), True)