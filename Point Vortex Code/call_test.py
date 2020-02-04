# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 15:08:18 2020

@author: Zak
"""
import pickle
import sys
import time

fname = f'test{sys.argv[1]}file.derp'
data = {'test': 55}
with open(fname, "wb") as f:
    pickle.dump(data, f)
    
time.sleep(30)