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

class InverseDensity(RiemannianManifold):
    #Probability Density Geodesics in Image Diffusion Latent Space
    def __init__(self,
                 log_prob:Callable[[torch.Tensor], torch.Tensor],
                 dim:int=2,
                 )->None:

        self.dim = dim

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

        val = 1./(torch.exp(log_prob_fn(x_samples))**2)
        I = torch.eye(self.dim, device=x_samples.device)            # shape (d, d)
        if x_samples.ndim > 1:
            I_batch = I.unsqueeze(0).repeat(*x_samples.shape[:-1], 1, 1)  # shape (n, d, d)
        else:
            I_batch = I

        return val*I_batch