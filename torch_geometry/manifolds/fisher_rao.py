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

class FisherRao(RiemannianManifold):
    def __init__(self,
                 log_prob:Callable[[torch.Tensor], torch.Tensor],
                 dim:int=2,
                 lam:float=1e-2,
                 )->None:

        self.dim = dim
        self.lam = lam

        super().__init__(G=lambda x: self.fisher_rao_metric(log_prob, x), f=lambda x: x, invf= lambda x: x)
        
        return
    
    def __str__(self)->str:
        
        return f"Euclidean manifold of dimension {self.dim} in standard coordinates"
    
    def fisher_rao_metric(self, log_prob_fn, x_samples):
        """
        Compute Fisher-Rao metric efficiently using vectorization.
        
        Args:
            log_prob_fn: θ -> scalar log-prob
            x_samples: (N, d) tensor of samples
        Returns:
            G: (d, d) Fisher-Rao metric
        """
        # Score function
        score_fn = jacrev(log_prob_fn)  # θ -> grad (d,)
        
        # Batch computation
        scores = score_fn(x_samples)  # (N, d)
        
        G = torch.einsum('...i,...j->...ij', scores, scores)
        
        identity = torch.eye(self.dim, device=x_samples.device)*self.lam

        return (G + identity).squeeze()