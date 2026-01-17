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

class HyperbolicParaboloid(RiemannianManifold):
    def __init__(self,
                 )->None:

        self.dim = 2
        self.emb_dim = 3
        super().__init__(f=self.f_standard, invf=self.invf_standard)
        
        return
    
    def __str__(self)->str:
        
        return "Hyperbolic Paraboloid equipped with the pull back metric"
    
    def f_standard(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (..., 2)
        returns (..., 3)
        """
        z0 = z[..., 0]
        z1 = z[..., 1]
        extra = z0**2 - z1**2                        # (...,)
        return torch.cat([z, extra[..., None]], dim=-1)

    def invf_standard(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., 3)
        returns (..., 2)
        """
        return x[..., :-1]
        