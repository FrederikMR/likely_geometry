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

#%% Code

class nEllipsoid(RiemannianManifold):
    def __init__(self,
                 dim:int=2,
                 params:Tensor=None,
                 coordinates="stereographic",
                 )->None:
        
        if params == None:
            params = torch.ones(dim+1, dtype=torch.float32)
        self.params = params
        self.coordinates = coordinates
        if coordinates == "stereographic":
            f = self.f_stereographic
            invf = self.invf_stereographic
        elif coordinates == "spherical":
            f = self.f_spherical
            invf = self.invf_spherical
        else:
            raise ValueError(f"Invalid coordinate system, {coordinates}. Choose either: \n\t-stereographic\n\t-spherical")
        
        self.dim = dim
        self.emb_dim = dim +1
        super().__init__(f=f, invf=invf)
        
        return
    
    def __str__(self)->str:
        
        return f"Sphere of dimension {self.dim} in {self.coordinates} coordinates equipped with the pull back metric"
    
    def f_stereographic(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (..., d)
        return: (..., d+1)
        """
        s2 = torch.sum(z**2, dim=-1)                     # (...,)
        num_first = (1 - s2)[..., None]                 # (..., 1)
        num_rest = 2 * z                                # (..., d)
        num = torch.cat([num_first, num_rest], dim=-1)  # (..., d+1)
        denom = (1 + s2)[..., None]                     # (..., 1)
    
        return self.params * num / denom                # broadcast: (..., d+1)

    def invf_stereographic(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., d+1)
        return: (..., d)
        """
        x = x / self.params                # broadcast-safe
    
        x0 = x[..., 0]                     # (...,)
        rest = x[..., 1:]                  # (..., d)
    
        return rest / (1 + x0)[..., None]  # (..., d)
        
    def f_spherical(self, 
                    z:Tensor,
                    )->Tensor:
        
        sin_term = torch.sin(z)
        cos_term = torch.cos(z)
        
        xn = cos_term[0]
        xi = torch.cumprod(sin_term[:-1])*cos_term[1:]
        x1 = torch.prod(sin_term)
        
        return self.params*torch.hstack((x1, xi, xn))

    def invf_spherical(self, 
                       x:Tensor,
                       )->Tensor:
        
        x /= self.params
        
        cum_length = torch.sqrt(torch.cumsum(x[1::-1]**2))
        
        return vmap(lambda cum, x: torch.arctan2(cum, x))(cum_length, x[:-1])
    
    def dist(self,
             x:Tensor,
             y:Tensor
             )->Tensor:
        
        return torch.arccos(torch.dot(x,y))
    
    def Log(self,
            x:Tensor,
            y:Tensor,
            )->Tensor:
        
        x /= self.params
        y /= self.params
        
        dot = torch.dot(x,y)
        dist = self.dist(x,y)
        val = y-dot*x
        
        return self.params*dist*val/torch.linalg.norm(val)
    
    
    
    