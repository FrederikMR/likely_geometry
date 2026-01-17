#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 17:16:29 2025

@author: fmry
"""

#%% Modules

import torch
from torch import Tensor

#%% Energy function

def euclidean_energy(curve:Tensor):
    """Computes Euclidean energy for a discretized curve

    Args:
      curve: curve
    Output:
      Euclidean energy of the curve
    """

    u = curve[1:]-curve[:-1]

    return torch.sum(u**2)

#%% Energy function

def riemannian_energy(G, curve:Tensor):
    """Computes Euclidean energy for a discretized curve

    Args:
      curve: curve
    Output:
      Euclidean energy of the curve
    """
    
    Gi = torch.vmap(G)(curve[:-1])
    u = curve[1:]-curve[:-1]
    
    return torch.sum(torch.einsum('...i,...ij,...j->...', u, Gi, u))
