# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 10:42:32 2019

@author: Zak
"""

import PVM as pvm
import subprocess
import pickle


# Assumes the seed has been evolved

names =  [
    # 'N20_T10_S851125867',
#     'N30_T500_S144692810',
#     'N40_T500_S517932362'
        ]

# names = ['N50_T999_S840620860']

seeds = []

seedf = 'Metadata/N100_T999_Mixed.dat'
with open(seedf, 'rb') as f:
    seeds = pickle.load(f)
    
    
names = ['N100_T999_S' + str(s) for s in seeds]

for fname in names:
    analysis = pvm.Analysis(fname)
    
    to_analyze = [
    pvm.ANALYSIS_CHOICE.CLUSTER_ANALYSIS,
    pvm.ANALYSIS_CHOICE.AUTO_CORR_CLUSTER,
    pvm.ANALYSIS_CHOICE.RMS_CLUSTER_NON_CENTERED,
    pvm.ANALYSIS_CHOICE.RMS_CLUSTER_CENTERED,
    pvm.ANALYSIS_CHOICE.RMS_NON_DIPOLE_CENTERED,
    pvm.ANALYSIS_CHOICE.RMS_NON_DIPOLE_NON_CENTERED,
    pvm.ANALYSIS_CHOICE.PAIR_CORR_W,
    pvm.ANALYSIS_CHOICE.PAIR_CORR,
    pvm.ANALYSIS_CHOICE.DIPOLE_MOMENT
    ]
    
    analysis.run(to_analyze)
    analysis.save()
    
# If analysis complete, evolve & run new seeds.
    
# N = 200
# i = 0

# ## Note, subprocess returns only when process is complete. So we don't run 50 concurrent evolutions.
# while i < N:
#     subprocess.call('python run_dynamics_save.py')
#     print(f"Evolution + Analysis complete for the {i}th time.")
#     i = i + 1
