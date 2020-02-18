# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 14:57:03 2020

@author: zakla
"""

import PVM as pvm
import numpy as np
import pandas as pd
import pickle
import re
from os.path import join, isfile
from os import listdir

### Run to show information about a given seed. 

# Fix pandas number formatting(not scientific)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

### Seed to look at
seed = 920411951#428814872#890486217

# Enumerate all files
# Select all analysis with N50_T500 with mixed (even) signs
path = 'Datafiles/'
files = [f for f in listdir(path) if isfile(join(path, f))]

matches = []
for f in files:
    # Pick out only evolution files, then seeds are automatically unique 
    expr = f"E_N[0-9]+_T[0-9]+_S{seed}\.dat"
    expr2 = f"E_N[0-9]+_T[0-9]+_R[0-9]+G_[0-9]+.[0-9]+_S[0-9]+"
    m = re.search(expr, f)
    
    if m:
        matches.append(m[0])
    
for fname in matches:
    ef = open('Datafiles/' + fname, "rb")
    data = pickle.load(ef)
    
    df = pd.DataFrame([data['settings']]).T
    print(f"Settings for file {fname}\n")
    print(df)
    