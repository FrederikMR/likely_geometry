#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 01:05:15 2025

@author: fmry
"""

#%% Modules

import torch

from torch import Tensor

from abc import ABC

#%% Spherical Interpoalation

class SphericalInterpolation(ABC):
    """SphericalInterpolation performs spherical interpolation

    Attributes:
        N: number of grid points for output curve with N+1 grid points
    """
    def __init__(self,
                 N:int=100,
                 device:str=None,
                 )->None:
        """Initilization of Spherical Interpolation

        Args:
            N: number of grid points for output curve with N+1 grid points
        """
        
        self.N = N
        
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        return
    
    @torch.no_grad()
    def __call__(self,
                 z0:Tensor,
                 zN:Tensor,
                 )->Tensor:
        """Interpolates between two points

        Args:
          z0: start point of curve
          zN: end point of curve
        Output:
          connecting spherical interpolation between z0 and zN
        """
        
        shape = z0.shape
        
        z0 = z0.reshape(-1)
        zN = zN.reshape(-1)
        
        z0_norm = torch.linalg.norm(z0)
        zN_norm = torch.linalg.norm(zN)
        dot_product = torch.sum(z0*zN)/(z0_norm*zN_norm)
        dot_product = dot_product.clamp(-1.+1e-7, 1.-1e-7)
        theta = torch.arccos(dot_product)
        
        sin_theta = torch.sin(theta)
        
        s = torch.linspace(0,1,self.N+1, device=self.device)[1:-1].reshape(-1,1)
        
        curve = ((z0*torch.sin((1.-s)*theta) + zN*torch.sin(s*theta))/sin_theta)
        
        curve = torch.vstack((z0, curve, zN))
        
        return curve.reshape(-1, *shape)
