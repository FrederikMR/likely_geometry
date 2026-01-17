#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 15:00:51 2025

@author: fmry
"""

#%% Modules

from jax_geometry.setup import *

from jax_geometry.manifolds import RiemannianManifold

#%% PGA

class PGA(ABC):
    def __init__(self,
                 )->None:
        
        return
    
    def __str__(self,
                )->str:
        
        return "PGA class"
    
    def __call__(self,
                 z_mu:Array,
                 u0s:Array,
                 )->Array:
        
        pga_S = jnp.mean(jnp.einsum('...i,...j->...ij', u0s, u0s), axis=0)
        U, S, V = jnp.linalg.svd(pga_S)
        
        return V
        