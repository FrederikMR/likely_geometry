#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 10:51:29 2024

@author: fmry
"""

#%% Sources

#%% Modules

import torch

from torch import Tensor, vmap
from torch.func import jacrev

from typing import Callable

####################

from .manifold import RiemannianManifold

#%% Code

class Linear:
    def __init__(self,
                 eps:float=1e-8,
                 )->None:

        self.eps = eps
        
        return
    
    def __str__(self)->str:
        
        return f"Euclidean manifold of dimension {self.dim} in standard coordinates"
    
    def ivp_geodesic(self, x0, v, N_grid=100):
        
        return x0 + v*torch.linspace(0.,1.,N_grid+1, device=x0.device)[1:].reshape(-1,1)
    
    def bvp_geodesic(self, x1,x2, N_grid=100):
        
        return x1 + (x2 - x1)*torch.linspace(0.,1.,N_grid+1, device=x1.device).reshape(-1,1)
    
    def mean_com(self, data_sample, N_grid=100):
        
        mean = torch.mean(data_sample, axis=0)
        
        curves = torch.vmap(self.bvp_geodesic, in_dims=(None,0,None))(mean, data_sample, N_grid)
        
        return mean, curves