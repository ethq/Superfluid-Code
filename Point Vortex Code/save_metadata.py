# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 10:48:45 2020

@author: Zak

"""
from os import listdir
from os.path import isfile, join
# import pandas as pd
import re
import pickle
import numpy as np

# Enumerates all data analyzed & saves a list of relevant seeds/filenames satisfying given criteria


# Select all analysis with N50_T500 with mixed (even) signs
path = 'Datafiles/'
files = [f for f in listdir(path) if isfile(join(path, f))]

seeds = []
for f in files:
    # print(f)
    # Pick out only evolution files, then seeds are automatically unique
    expr = 'Evolution_N50_T500_S[0-9]+\.dat'
    m = re.search(expr, f)
    
    if not m:
        print(f'File not added: {f}')
        continue
    
    # Make sure its got an even number of signs and radius 200
    fname = 'Datafiles/' + f
    with open(fname, "rb") as g:
        data = pickle.load(g)  
    
    # Sum circulations
    c = int(np.sum(data['circulations'][0][0]))
    
    r = int(data['settings']['domain_radius'])
    
    if c or r != 200:
        print(f'File had incorrect circ({c}) or radius({r}). {f}')
        continue
    
    # Slice is as follows because Evolution_N50_T500_S = 20 characters, .dat = 4 characters
    seed = int(m[0][20:-4])
    
    seeds.append(seed)
    print(f'Added seed: {seed}')

fname = 'Metadata/N50_T500_Mixed.dat'
with open(fname, 'wb') as f:
    pickle.dump(seeds, f)
    
print('Analysis done')