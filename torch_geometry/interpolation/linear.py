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

class LinearInterpolation(ABC):
    """LinearInterpolation performs spherical interpolation

    Attributes:
        N: number of grid points for output curve with N+1 grid points
    """
    def __init__(self,
                 N:int=100,
                 device:str=None,
                 )->None:
        """Initilization of Linear Interpolation

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
          connecting linear interpolation between z0 and zN
        """
        
        shape = z0.shape
        
        z0 = z0.reshape(-1)
        zN = zN.reshape(-1)
        
        curve = (zN-z0)*torch.linspace(0.0,1.0,self.N+1,dtype=z0.dtype, device=self.device)[1:-1].reshape(-1,1)+z0
        
        curve = torch.vstack((z0, curve, zN))
        
        return curve.reshape(-1, *shape)
