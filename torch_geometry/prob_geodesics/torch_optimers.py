#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Torch Optimizer for Geodesic Estimation
"""

import torch
from torch import Tensor
from typing import Callable, Tuple
from abc import ABC


class TorchOptimizers_Euclidean(ABC):
    def __init__(
        self,
        reg_fun: Callable,
        init_fun: Callable = None,
        lam: float = 1.0,
        N: int = 100,
        tol: float = 1e-4,
        max_iter: int = 1000,
        optimizer_class=torch.optim.Adam,
        lr: float = 0.01,
        optimizer_kwargs=None,
        device: str = None,
    ) -> None:

        self.reg_fun = reg_fun
        self.lam = lam
        self.N = N
        self.tol = tol
        self.max_iter = max_iter
        self.lr = lr
        self.optimizer_class = optimizer_class

        if optimizer_kwargs is None:
            self.optimizer_kwargs = {}
        else:
            self.optimizer_kwargs = optimizer_kwargs

        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        if init_fun is None:
            self.init_fun = lambda z0, zN, N: (
                (zN - z0)
                * torch.linspace(0.0, 1.0, N + 1, dtype=z0.dtype, device=self.device)[
                    1:-1
                ].reshape(-1, 1)
                + z0
            )
        else:
            self.init_fun = init_fun

    def __str__(self) -> str:
        return "Geodesic Computation Object using Control Problem with Probability Flow"

    def initialize(self) -> Tensor:
        zi = self.init_fun(self.z0, self.zN, self.N)
        return zi

    def energy(self, z: Tensor) -> Tensor:
        """
        Energy functional: sum of squared differences
        z: (N-1, dim)
        """
        # Concatenate endpoints without breaking the graph
        zi = torch.cat([self.z0.unsqueeze(0), z, self.zN.unsqueeze(0)], dim=0)
        ui = zi[1:] - zi[:-1]
        return torch.sum(ui * ui)

    def reg_energy(self, z: Tensor) -> Tensor:
        """
        Regularized energy: E + lambda * reg
        """
        # Ensure reg_fun does not modify z in-place
        reg_val = self.reg_fun(z)
        energy = self.energy(z)
        return energy + self.lam_norm * reg_val

    def __call__(self, z0: Tensor, zN: Tensor) -> Tensor:
        """
        Compute the geodesic between z0 and zN.
        """
        shape = z0.shape
        device = z0.device

        # Store endpoints (no detach, to preserve graph)
        self.z0 = z0.reshape(-1).detach().to(device)
        self.zN = zN.reshape(-1).detach().to(device)
        self.dim = len(z0)

        # Initialize internal points
        zi = self.initialize().to(device)
        zi = zi.clone().detach().requires_grad_(True)

        # Normalize lambda
        energy_init = self.energy(zi).detach()
        reg_val_init = torch.abs(self.reg_fun(zi).detach())
        if reg_val_init > 1e-6:
            self.lam_norm = self.lam * energy_init / reg_val_init
        else:
            self.lam_norm = self.lam

        # Make zi a trainable parameter
        z = torch.nn.Parameter(zi)

        # Instantiate optimizer
        optimizer = self.optimizer_class([z], lr=self.lr, **self.optimizer_kwargs)

        for i in range(self.max_iter):
            optimizer.zero_grad()

            # Recompute the graph each iteration
            loss = self.reg_energy(z)
            loss.backward()

            grad_norm = z.grad.norm()

            # Step optimizer
            if isinstance(optimizer, torch.optim.LBFGS):
                # LBFGS requires closure
                def closure():
                    optimizer.zero_grad()
                    l = self.reg_energy(z)
                    l.backward()
                    return l

                optimizer.step(closure)
            else:
                optimizer.step()

            # Optional stopping criterion
            if grad_norm < self.tol:
                break
            
        z_full = torch.cat(
            [
                self.z0.detach().reshape(1,-1),
                z.detach(),
                self.zN.detach().reshape(1,-1),
                ],
            dim=0
            )
    
        return z_full.reshape(-1, *shape)
