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

class SphericalInterpolation(ABC):
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
        
        z0_norm = jnp.linalg.norm(z0)
        zN_norm = jnp.linalg.norm(zN)
        dot_product = jnp.dot(z0, zN)
        theta = jnp.arccos(dot_product/(z0_norm*zN_norm))
        
        sin_theta = jnp.sin(theta)
        
        i = jnp.linspace(0.0,1.0,self.T,endpoint=False,dtype=z0.dtype)[1:].reshape(-1,1)
        
        curve = ((z0*jnp.sin((1.-i)*theta) + zN*jnp.sin(i*theta))/sin_theta)
        
        curve = jnp.vstack((z0, curve, zN))
        
        return curve.reshape(-1, *shape)