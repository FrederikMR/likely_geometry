#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 10:51:29 2024

@author: fmry
"""

#%% Sources

#%% Modules

from torch import Tensor
from typing import Callable

####################

from .manifold import RiemannianManifold

#%% Code

class LatentSpaceManifold(RiemannianManifold):
    def __init__(self,
                 dim,
                 emb_dim,
                 encoder:Callable[[Tensor], Tensor],
                 decoder:Callable[[Tensor], Tensor],
                 )->None:

        self.dim = dim
        self.emb_dim = emb_dim
        super().__init__(f=decoder, invf=encoder)
        
        return