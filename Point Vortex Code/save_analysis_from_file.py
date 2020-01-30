# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 10:42:32 2019

@author: Zak
"""

import PVM as pvm

# Assumes the seed has been evolved

names =  [
        'N20_T50_S768390681',
        'N20_T50_S457173602',
        'N20_T50_S869893185',
        'N26_T50_S717109192',
        'N10_T50_S87655771',
        'N10_T50_S996866482',
        'N10_T50_S873349814'
        ]

for fname in names:
    analysis = pvm.Analysis(fname)
    analysis.full_analysis()
    analysis.save()