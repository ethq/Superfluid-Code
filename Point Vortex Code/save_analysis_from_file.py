# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 10:42:32 2019

@author: Zak
"""

import PVM as pvm
import subprocess


# Assumes the seed has been evolved

# names =  [
#     'N20_T500_S402701135',
#     'N30_T500_S144692810',
#     'N40_T500_S517932362'
#         ]

# for fname in names:
#     analysis = pvm.Analysis(fname)
#     analysis.full_analysis()
#     analysis.save()
    
# If analysis complete, evolve & run new seeds.
    
N = 50
i = 0

## Note, subprocess returns only when process is complete. So we don't run 50 concurrent evolutions.
while i < N:
    subprocess.call('python run_dynamics_save.py')
    print(f"Evolution + Analysis complete for the {i}th time.")
    i = i + 1
