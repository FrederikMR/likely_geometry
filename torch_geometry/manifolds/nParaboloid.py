#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 10:51:29 2024

@author: fmry
"""

#%% Sources

#%% Modules

import torch

from torch import Tensor

####################

from .manifold import RiemannianManifold

#%% Code

class nParaboloid(RiemannianManifold):
    def __init__(self,
                 dim:int=2,
                 )->None:

        self.dim = dim
        self.emb_dim = dim+1
        super().__init__(f=self.f_standard, invf=self.invf_standard)
        
        return
    
    def __str__(self)->str:
        
        return f"Paraboloid of dimension {self.dim} equipped with the pull back metric"
    
    def f_standard(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (..., d)
        returns (..., d+1)
        """
        s2 = torch.sum(z**2, dim=-1, keepdim=True)   # (..., 1)
        return torch.cat([z, s2], dim=-1)            # (..., d+1)

    def invf_standard(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., d+1)
        returns (..., d)
        """
        return x[..., :-1]
        