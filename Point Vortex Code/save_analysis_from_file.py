# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 10:42:32 2019

@author: Zak
"""

import PVM as pvm

# Assumes the seed has been evolved

fname = 'N20_T10_S18817436'
analysis = pvm.Analysis(fname)
analysis.full_analysis()
analysis.save()