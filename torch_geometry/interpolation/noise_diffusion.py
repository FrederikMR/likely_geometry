#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 01:05:15 2025

@author: fmry
"""

#%% Modules

import torch
from torch import vmap

from torch import Tensor
from typing import Callable

from abc import ABC

#%% Spherical Interpoalation

class NoiseDiffusion(ABC):
    """NoiseDiffusion performs spherical interpolation with noise

    Attributes:
        N: number of grid points for output curve with N+1 grid points
        boundary: clips all variables between -/+ boundary
        sigma: noise level
        alpha: function that determines interpolation curve
        beta: function that determines interpolation curve
        gamma: function that determines interpolation curve
        mu: function that determines interpolation curve
        nu: function that determines interpolation curve
    """

    def __init__(self,
                 N:int=100,
                 boundary:float=2.0,
                 sigma:float=1.0,
                 alpha:Callable=lambda s: torch.cos(0.5*torch.pi*s),
                 beta:Callable=lambda s: torch.sin(0.5*torch.pi*s),
                 gamma:Callable|None= lambda s: 0.0,
                 mu:Callable|None= lambda s: None,
                 nu:Callable|None= lambda s: None,
                 device:str=None,
                 )->None:
        """Initilization of NoiseDiffusion

        Args:
            N: number of grid points for output curve with N+1 grid points
            boundary: clips all variables between -/+ boundary
            sigma: noise level
            alpha: function that determines interpolation curve
            beta: function that determines interpolation curve
            gamma: function that determines interpolation curve
            mu: function that determines interpolation curve
            nu: function that determines interpolation curve
        """
        
        self.N = N
        
        self.sigma = sigma
        self.boundary = boundary

        self.alpha = alpha
        self.beta = beta

        self.gamma = gamma if gamma is None else lambda s: torch.sqrt(torch.clip(1.-(self.alpha(s)**2)-(self.beta(s)**2), min=0.0, max=1.0))
        self.mu = mu if mu is None else lambda s: 1.2*self.alpha(s)/(self.alpha(s)+self.beta(s))
        self.nu = nu if nu is None else lambda s: 1.2*self.beta(s)/(self.alpha(s)+self.beta(s))
        
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        return
    
    @torch.no_grad()
    def __call__(self,
                 z0:Tensor,
                 zN:Tensor,
                 x0:Tensor,
                 xN:Tensor,
                 )->Tensor:
        """Interpolates between two points

        Args:
          z0: start point of curve in noise space
          zN: end point of curve in noise space
          x0: point in image space with f(x0)=z0, where f encodes into noise space
          xN: point in image space with f(xN)=zN, where f encodes into noise space
        Output:
          connecting linear interpolation between z0 and zN
        """
        
        shape = z0.shape
        
        z0 = z0.reshape(-1)
        zN = zN.reshape(-1)
        x0 = x0.reshape(-1)
        xN = xN.reshape(-1)
         
        z0 = torch.clip(z0, min=-self.boundary, max=self.boundary).reshape(-1)
        zN = torch.clip(zN, min=-self.boundary, max=self.boundary).reshape(-1)
        
        s = torch.linspace(0,1,self.N+1, device=self.device)[1:-1].reshape(-1,1) 
        
        alpha = vmap(self.alpha)(s)
        beta = vmap(self.beta)(s)
        gamma = vmap(self.gamma)(s)
        mu = vmap(self.mu)(s)
        nu = vmap(self.nu)(s)
        eps = self.sigma*torch.randn_like(z0)
        
        curve = alpha*z0+beta*zN+(mu-alpha)*x0+(nu-beta)*xN+gamma*eps
        curve = torch.vstack((z0, curve, zN))
        
        curve = torch.clip(curve, -self.boundary, self.boundary)
        
        return curve.reshape(-1, *shape)