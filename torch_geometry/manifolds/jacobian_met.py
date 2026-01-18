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
from torch.func import jacfwd

from typing import Callable

####################

from .manifold import RiemannianManifold

#%% Code

class JacobianMetric(RiemannianManifold):
    def __init__(self,
                 log_prob:Callable[[torch.Tensor], torch.Tensor],
                 dim:int=2,
                 lam:float=1e-2,
                 )->None:

        self.dim = dim
        self.lam = lam

        super().__init__(G=lambda x: self.fisher_rao_jtj(log_prob, x), f=lambda x: x, invf= lambda x: x)
        
        return
    
    def __str__(self)->str:
        
        return f"Euclidean manifold of dimension {self.dim} in standard coordinates"

    def fisher_rao_jtj(self, log_prob_fn, x_samples):
        """
        Compute Fisher-Rao metric as J^T J of the score function.
        """
        
        shape = None
        if x_samples.ndim == 1:
            x_samples = x_samples.reshape(-1,self.dim)
        elif x_samples.ndim == 3:
            shape = x_samples.shape[:-1]
            x_samples = x_samples.reshape(-1,self.dim)
            
        # Hessian per sample
        score_fn = jacfwd(log_prob_fn, randomness="same")   # θ -> d
        hess_fn  = jacfwd(score_fn, randomness="same")             # θ -> d x d
        
        Hs = torch.zeros((len(x_samples), self.dim, self.dim), device=x_samples.device)

        for i in range(len(x_samples)):
            Hs[i] = hess_fn(x_samples[i])
        
        # Compute Fisher-Rao G = J^T J
        G = torch.einsum('...ij,...ik->...jk', Hs, Hs)
        
        if shape is not None:
            G = G.reshape(*shape, self.dim, self.dim)
        # Add small identity for numerical stability
        
        identity = torch.eye(self.dim, device=x_samples.device)*self.lam
        
        return (G + identity).squeeze()