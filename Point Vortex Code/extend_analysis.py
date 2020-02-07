# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 10:29:57 2020

@author: Zak
"""

import numpy as np
import pickle
import PVM as pvm

# Choose seeds to extend analysis on
seeds = []

seedf = 'Metadata/N50_T500_Mixed.dat'
with open(seedf, 'rb') as f:
    seeds = pickle.load(f)
    
    
fnames = ['N50_T500_S' + str(s) for s in seeds]

for f in fnames:
    a = pvm.Analysis()
    
    achoice = [pvm.ANALYSIS_CHOICE.AUTO_CORR_CLUSTER, 
               pvm.ANALYSIS_CHOICE.AUTO_CORR_NON_DIPOLE
               ]
    a.extend(f, achoice)
    
    break