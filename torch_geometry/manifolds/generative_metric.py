#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 10:51:29 2024

@author: fmry
"""

#%% Sources

#%% Modules

import torch

from typing import Callable

####################

from .manifold import RiemannianManifold

#%% Code

class GenerativeMetric(RiemannianManifold):
    #https://arxiv.org/pdf/2407.11244
    def __init__(self,
                 log_prob:Callable[[torch.Tensor], torch.Tensor],
                 dim:int=2,
                 lam:float=1.0,
                 p0:float=1.0,
                 )->None:

        self.dim = dim
        self.lam = lam
        self.p0 = p0

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
        
        
        val = ((self.p0+self.lam) / (torch.exp(log_prob_fn(x_samples)) + self.lam))**2

        I = torch.eye(self.dim, device=x_samples.device)            # shape (d, d)
        if x_samples.ndim > 1:
            I_batch = I.unsqueeze(0).repeat(*x_samples.shape[:-1], 1, 1)  # shape (n, d, d)
        else:
            I_batch = I

        return val*I_batch