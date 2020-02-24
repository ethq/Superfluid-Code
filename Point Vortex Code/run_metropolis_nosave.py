# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 11:18:42 2020

@author: Zak
"""

import PVM as pvm
import numpy as np
from tqdm import tqdm
    
N = 20
R = 2

cfg0 = pvm.Configuration(
        N,
        R,
        pvm.CONFIG_STRAT.SINGLE_CLUSTER,
        pvm.CONFIG_STRAT.CIRCS_EVEN,
        None,
        None,
        {
                'minimum_separation': 1e-2
        }
        )

cfg = {
       'n_vortices': N,
       'pos': cfg0.pos,
       'circs': cfg0.circulations[0],
       'temperature': 1e3,
       'bbox_ratio': 10,
       'vorticity_tol': 1e-3,
       'annihilation_threshold': 1e-2,
       'domain_radius': R,
       'skip': 1,
       'total_steps': 5000
       }
acc = 0
for i in tqdm(1 + np.arange(100)):
    e = pvm.Evolver_MCMC(**cfg)
    e.evolve()
    acc = acc + e.accepted_rel
    tqdm.write(f"Accepted: {acc/i} %")
print(f"Accepted: {acc/100} %")

p = pvm.HarryPlotter()


def pcfg():
    cfg = {'positions': np.concatenate([e.pos, e.impos]), 'circulations': np.concatenate([e.circs, -1*e.circs])}
    p.plot_cfg(cfg)
    
pcfg()