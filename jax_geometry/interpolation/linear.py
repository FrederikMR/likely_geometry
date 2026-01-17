#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 01:05:15 2025

@author: fmry
"""

#%% Modules

import jax.numpy as jnp

from jax import Array
from abc import ABC

#%% Spherical Interpoalation

class LinearInterpolation(ABC):
    def __init__(self,
                 N:int=100,
                 )->None:
        
        self.N = N
        
        return
    
    def __call__(self,
                 z0:Array,
                 zN:Array,
                 )->Array:
        
        shape = z0.shape
        
        z0 = z0.reshape(-1)
        zN = zN.reshape(-1)
        
        curve = (zN-z0)*jnp.linspace(0.0,1.0,self.N,endpoint=False,dtype=z0.dtype)[1:].reshape(-1,1)+z0
        curve = jnp.vstack((z0, curve, zN))
        
        return curve.reshape(-1, *shape)