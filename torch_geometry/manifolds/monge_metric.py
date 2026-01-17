#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 10:51:29 2024

@author: fmry
"""

#%% Sources

#%% Modules

import torch

from torch.func import jacrev

from typing import Callable

####################

from .manifold import RiemannianManifold

#%% Code

class MongeMetric(RiemannianManifold):
    #Lagrangian manifold Monte Carlo on Monge patches
    def __init__(self,
                 log_prob:Callable[[torch.Tensor], torch.Tensor],
                 dim:int=2,
                 alpha:float=1.0,
                 )->None:

        self.dim = dim
        self.alpha = alpha

        super().__init__(G=lambda x: self.metric(log_prob, x), f=lambda x: x, invf= lambda x: x)
        
        return
    
    def __str__(self)->str:
        
        return f"Euclidean manifold of dimension {self.dim} in standard coordinates"
    
    def metric(self, log_prob_fn, x_samples):
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
        
        val = torch.einsum('...i,...i->...', scores, scores)
        
        I = torch.eye(self.dim, device=x_samples.device)            # shape (d, d)
        if x_samples.ndim > 1:
            I_batch = I.unsqueeze(0).repeat(*x_samples.shape[:-1], 1, 1)  # shape (n, d, d)
        else:
            I_batch = I

        if x_samples.ndim > 1:
            return (I_batch + self.alpha * val[...,None,None] * I_batch).squeeze()
        else:
            return (I_batch + self.alpha * val * I_batch).squeeze()