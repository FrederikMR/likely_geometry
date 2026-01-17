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

class Spherical:
    def __init__(self,
                 eps:float=1e-8,
                 )->None:

        self.eps = eps
        
        return
    
    def __str__(self)->str:
        
        return f"Euclidean manifold of dimension {self.dim} in standard coordinates"
    
    def ivp_geodesic(self, x0, v, N_grid=100):
        device = x0.device
        eps = self.eps
    
        t = torch.linspace(0., 1., N_grid + 1, device=device)
        t = t.view(-1, *([1] * x0.dim()))
    
        # Speed = angular magnitude
        theta = torch.norm(v, dim=-1, keepdim=True).clamp(min=eps)
    
        # Construct endpoint implied by IVP
        v_dir = v / theta
        x1 = torch.cos(theta) * x0 + torch.sin(theta) * v_dir
    
        # SLERP between x0 and x1 (Section 3.2 form)
        sin_theta = torch.sin(theta)
    
        s1 = torch.sin((1 - t) * theta) / (sin_theta + eps)
        s2 = torch.sin(t * theta) / (sin_theta + eps)
    
        gamma = s1 * x0 + s2 * x1
        return gamma
    
    def bvp_geodesic(self, x1, x2, N_grid=100):
        device = x1.device
        eps = self.eps
    
        # --------------------------------------------------
        # Interpolation parameter
        # --------------------------------------------------
        t = torch.linspace(0., 1., N_grid + 1, device=device)
        t = t.view(-1, *([1] * x1.dim()))  # (N_grid+1, ..., 1)
    
        # --------------------------------------------------
        # Angle computation (THIS is the crucial fix)
        # --------------------------------------------------
        r1 = x1.norm(dim=-1, keepdim=True)
        r2 = x2.norm(dim=-1, keepdim=True)
    
        cos_theta = (x1 * x2).sum(dim=-1, keepdim=True) / (r1 * r2 + eps)
        cos_theta = cos_theta.clamp(-1 + eps, 1 - eps)
    
        theta = torch.acos(cos_theta)
        sin_theta = torch.sin(theta)
    
        # --------------------------------------------------
        # SLERP weights (exactly as in Section 3.2)
        # --------------------------------------------------
        s1 = torch.sin((1 - t) * theta) / (sin_theta + eps)
        s2 = torch.sin(t * theta) / (sin_theta + eps)
    
        slerp = s1 * x1 + s2 * x2
    
        # --------------------------------------------------
        # LERP fallback for small angles
        # --------------------------------------------------
        lerp = (1 - t) * x1 + t * x2
    
        small_angle = sin_theta < eps
        while small_angle.dim() < slerp.dim():
            small_angle = small_angle.unsqueeze(0)
    
        out = torch.where(small_angle, lerp, slerp)
        return out


    def sphere_log(self, x, y):
        dot = (x * y).sum(dim=-1, keepdim=True).clamp(-1 + self.eps, 1 - self.eps)
        theta = torch.acos(dot)
        v = y - dot * x
        norm_v = torch.norm(v, dim=-1, keepdim=True).clamp(min=self.eps)
        return v * (theta / norm_v)
    
    def sphere_exp(self, x, v):
        norm_v = torch.norm(v, dim=-1, keepdim=True).clamp(min=self.eps)
        theta = norm_v
        return torch.cos(theta) * x + torch.sin(theta) * (v / norm_v)
    
    def mean_com(self, points, n_iter=100, lr=0.5, N_grid=100):
        """
        Compute a “radius-aware” Karcher mean of points in R^n using Section 3.2 SLERP.
    
        Steps:
            1. Normalize points to the unit sphere
            2. Compute the spherical Karcher mean
            3. Scale mean to the average radius of the original data
            4. Use Section 3.2 SLERP to interpolate between mean and points
    
        Args:
            points: (N, D) tensor of points in R^n (can have arbitrary norm)
            n_iter: number of Riemannian gradient descent steps for Karcher mean
            lr: step size for gradient descent
            N_grid: number of points along the SLERP curves
    
        Returns:
            mean_scaled: (D,) the mean pushed to average radius
            curves: (N_grid+1, N, D) interpolated curves from mean to each point
        """
    
        # ------------------------------------------------------
        # 1️⃣ Normalize all points to unit sphere
        # ------------------------------------------------------
        points_norm = points / points.norm(dim=-1, keepdim=True)
    
        # ------------------------------------------------------
        # 2️⃣ Compute Karcher mean on the unit sphere
        # ------------------------------------------------------
        mean_unit = points_norm[0].clone()  # initialize
        for _ in range(n_iter):
            log_sum = self.sphere_log(mean_unit.unsqueeze(0), points_norm).mean(dim=0)
            mean_unit = self.sphere_exp(mean_unit, -lr * log_sum)
            mean_unit = mean_unit / mean_unit.norm()  # numerical safety
    
        # ------------------------------------------------------
        # 3️⃣ Push Karcher mean out to average radius
        # ------------------------------------------------------
        avg_radius = points.norm(dim=-1).mean()
        mean_scaled = mean_unit * avg_radius
    
        # ------------------------------------------------------
        # 4️⃣ Compute Section 3.2 SLERP curves
        # ------------------------------------------------------
        curves = torch.vmap(self.bvp_geodesic, in_dims=(None, 0, None))(
            mean_scaled, points, N_grid
        )
    
        return mean_scaled, curves
