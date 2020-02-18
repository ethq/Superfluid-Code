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

T = 5015    # Time of simulation
N = 50      # Number of vortices
R = 2000    # Domain radius
G = 0.3     # Rate of dissipation

seeds = []
for f in files:
    # print(f)
    # Pick out only evolution files, then seeds are automatically unique
    # expr = f"Evolution_N{N}_T{T}_S[0-9]+"
    # expr = f"E_N[0-9]+_T[0-9]+_R[0-9]+_G[0-9]+\.[0-9]+_S[0-9]+"
    expr = f"E_N{N}_T{T}_R{R}_G{G}_S[0-9]+"
    m = re.search(expr, f)
    
    if not m:
        # print(f'File not added: {f}')
        continue
    
    # # Make sure its got an even number of signs and radius 200
    # fname = 'Datafiles/' + f
    # with open(fname, "rb") as g:
    #     data = pickle.load(g)  
    
    # # Sum circulations
    # c = int(np.sum(data['circulations'][0][0]))
    
    # r = int(data['settings']['domain_radius'])
    
    # if not c or r != 2000:
    #     print(f'File had incorrect circ({c}) or radius({r}). {f}')
    #     continue
    print(f)
     # Slicing to extract seed
    nla = len(str(N)) + len(str(T))
    nlb = nla + len(str(R)) + len(str(G))
    # Slice is as follows because E_N_T_S = 7 characters
    sa = 7
    # Slice is as follows because E_N_T_R_G_S = 11 characters
    sb = 11
    
    seed = int(m[0][sb+nlb:])
    
    seeds.append(seed)
    print(f'Added seed: {m[0]}, s: {seed}')

fname = f"Metadata/N{N}_T{T}_Mixed.dat"
with open(fname, 'wb') as f:
    pickle.dump(seeds, f)
    
# print(seeds)
print('Analysis done')