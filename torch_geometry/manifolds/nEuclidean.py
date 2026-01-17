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

class nEuclidean(RiemannianManifold):
    def __init__(self,
                 dim:int=2,
                 )->None:

        self.dim = dim
        super().__init__(G=self.metric, f=lambda x: x, invf= lambda x: x)
        
        return
    
    def __str__(self)->str:
        
        return f"Euclidean manifold of dimension {self.dim} in standard coordinates"
    
    def metric(self,
               z:Tensor,
               )->Tensor:
        
        if z.ndim == 1:
            return torch.eye(self.dim, device=z.device)
        else:            
            diag = torch.eye(self.dim, device=z.device)
        
            diag_multi = z.unsqueeze(2).expand(*z.size(), z.size(1))
            diag3d = diag_multi*diag
        
            return diag3d
    
    def dist(self,
             z1:Tensor,
             z2:Tensor
             )->Tensor:
        
        return torch.linalg.norm(z2-z1, axis=-1)
    
    def Geodesic(self,
                 x:Tensor,
                 y:Tensor,
                 t_grid:Tensor=None,
                 )->Tensor:
        
        if t_grid is None:
            t_grid = torch.linspace(0.,1.,100)
        
        return x+(y-x)*t_grid.reshape(-1,1)
    