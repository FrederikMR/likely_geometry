#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 10:51:29 2024

@author: fmry
"""

#%% Sources

#%% Modules

import torch

from torch import vmap

from torch import Tensor

####################

from .manifold import RiemannianManifold
from .nEllipsoid import nEllipsoid

#%% Code

class nSphere(nEllipsoid):
    def __init__(self,
                 dim:int=2,
                 coordinates="stereographic",
                 )->None:
        super().__init__(dim=dim, params=torch.ones(dim+1, dtype=torch.float32), coordinates=coordinates)
        
        return
    
    def __str__(self)->str:
        
        return f"Sphere of dimension {self.dim} in {self.coordinates} coordinates equipped with the pull back metric"
    
    def Exp(self,
            x:Tensor,
            v:Tensor,
            t:float=1.0,
            )->Tensor:
        
        norm = torch.linalg.norm(v)
        
        return (torch.cos(norm*t)*x+torch.sin(norm*t)*v/norm)*self.params
    
    def Geodesic(self,
                 x:Tensor,
                 y:Tensor,
                 t_grid:Tensor=None,
                 )->Tensor:
        
        if t_grid is None:
            t_grid = torch.linspace(0.,1.,99, endpoint=False)[1:]
        
        x = self.f(x)
        y = self.f(y)
        
        x_s = x/self.params
        y_s = y/self.params
        
        v = self.Log(x,y)
        
        gamma = self.params*vmap(lambda t: self.Exp(x_s, v,t))(t_grid)
        
        return torch.vstack((x,gamma,y))
    
    
    