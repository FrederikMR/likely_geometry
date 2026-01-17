#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 12:01:26 2024

@author: fmry
"""

#%% Sources

#%% Modules

import torch
from torch import vmap
from torch.func import jacfwd, grad

from torch import Tensor
from typing import Callable, Dict, Tuple
from abc import ABC

from torch_geometry.manifolds import RiemannianManifold
from torch_geometry.line_search import Backtracking


#%% Gradient Descent Estimation of Geodesics

class ProbScoreGEORCE(ABC):
    def __init__(self,
                 M:RiemannianManifold,
                 score_fun:Callable,
                 init_fun:Callable=None,
                 lam:float=1.0,
                 N:int=100,
                 tol:float=1e-4,
                 max_iter:int=1000,
                 lr_rate:float=0.1,
                 beta1:float=0.5,
                 beta2:float=0.5,
                 eps:float=1e-8,
                 device:str=None,
                 )->None:
        
        self.M = M
        self.score_fun = score_fun
        
        self.lam = lam
        self.N = N
        self.tol = tol
        self.max_iter = max_iter

        self.lr_rate=lr_rate
        self.beta1=beta1
        self.beta2=beta2
        self.eps=eps
        
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        if init_fun is None:
            self.init_fun = lambda z0, zN, N: (zN-z0)*torch.linspace(0.0,
                                                                     1.0,
                                                                     N+1,
                                                                     dtype=z0.dtype,
                                                                     device=self.device)[1:-1].reshape(-1,1)+z0
        else:   
            self.init_fun = init_fun
        
    def __str__(self)->str:
        
        return "Geodesic Computation Object using Control Problem with Probability Flow"
    
    def initialize(self,
                   )->Tuple:
        
        zi = self.init_fun(self.z0,self.zN,self.N)
        
        val =  torch.vstack((self.z0, zi, self.zN))
        ui = val[1:]-val[:-1]
        
        return zi, ui
    
    @torch.no_grad()
    def energy(self, 
               zi:torch.Tensor,
               )->torch.Tensor:
        
        dz0 = zi[0]-self.z0
        e1 = torch.einsum('i,ij,j->', dz0, self.G0, dz0)
        
        zi = torch.vstack((zi, self.zN))
        Gi = self.M.G(zi[:-1])
        dzi = zi[1:]-zi[:-1]
        
        return e1+torch.sum(torch.einsum('...i,...ij,...j->...', dzi, Gi, dzi))
    
    @torch.no_grad()
    def Dregenergy(self,
                   zi:torch.Tensor,
                   ui:torch.Tensor,
                   Gi:torch.Tensor,
                   gi:torch.Tensor,
                   )->torch.Tensor:
        
        return gi+2.*(torch.einsum('tij,tj->ti', Gi[:-1], ui[:-1])-torch.einsum('tij,tj->ti', Gi[1:], ui[1:]))

    def inner_product(self,
                      zi:torch.Tensor,
                      ui:torch.Tensor,
                      )->torch.Tensor:
        
        Gi = self.M.G(zi)
        
        return torch.sum(torch.einsum('...i,...ij,...j->...', ui, Gi, ui)), Gi

    def gi(self,
           zi:torch.Tensor,
           ui:torch.Tensor,
           )->Tuple[torch.Tensor]:
        
        gi, Gi = grad(self.inner_product, has_aux=True)(zi,ui[1:])
        score_val = self.lam_norm*self.score_fun(zi)
        gi += score_val
        Gi = torch.vstack((self.G0.reshape(-1,self.M.dim,self.M.dim),
                           Gi))
        rg = torch.sum(gi**2)
        
        return gi, Gi, rg
    
    @torch.no_grad()
    def update_scheme(self, 
                      gi:torch.Tensor, 
                      Gi_inv:torch.Tensor,
                      )->torch.Tensor:
        
        g_cumsum = torch.flip(torch.cumsum(torch.flip(gi, dims=[0]), dim=0), dims=[0])
        ginv_sum = torch.sum(Gi_inv, axis=0)
        
        rhs = torch.sum(torch.einsum('tij,tj->ti', Gi_inv[:-1], g_cumsum), axis=0)+2.0*self.diff
        
        muN = -torch.linalg.solve(ginv_sum, rhs)
        mui = torch.vstack((muN+g_cumsum, muN))
        
        return mui
    
    @torch.no_grad()
    def update_xi(self,
                  zi:torch.Tensor,
                  alpha:torch.Tensor,
                  ui_hat:torch.Tensor,
                  ui:torch.Tensor,
                  )->torch.Tensor:
        
        return self.z0+torch.cumsum(alpha*ui_hat[:-1]+(1.-alpha)*ui[:-1], axis=0)
    
    @torch.no_grad()
    def update_step(self,
                    zi:torch.Tensor,
                    ui:torch.Tensor,
                    gi_hat:torch.Tensor,
                    Gi_inv:torch.Tensor,
                    kappa:float,
                    )->Tuple[torch.Tensor, torch.Tensor]:
        
        mui = self.update_scheme(gi_hat, Gi_inv)
        ui_hat = -0.5*torch.einsum('tij,tj->ti', Gi_inv, mui)
        zi_hat = self.z0+torch.cumsum(ui_hat[:-1], axis=0)

        return zi+kappa*(zi_hat-zi), ui+kappa*(ui_hat-ui)
    
    @torch.no_grad()
    def adaptive_update(self,
                        Gi_k1:torch.Tensor,
                        Gi_k2:torch.Tensor,
                        gi_k1:torch.Tensor,
                        gi_k2:torch.Tensor,
                        rg_k1:torch.Tensor,
                        rg_k2:torch.Tensor,
                        beta1:torch.Tensor,
                        beta2:torch.Tensor,
                        idx:int,
                        )->Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                 torch.Tensor, torch.Tensor, torch.Tensor]:
    
        Gi_k2 = (1.-self.beta1)*Gi_k2+self.beta1*Gi_k1
        gi_k2 = (1.-self.beta1)*gi_k2+self.beta1*gi_k1
        rg_k2 = (1.-self.beta2)*rg_k2 +self.beta2*rg_k1
        
        beta1 = beta1*self.beta1
        beta2 = beta2*self.beta2

        Gi_hat = Gi_k2/(1.-beta1)
        gi_hat = gi_k2/(1.-beta1)
        vt = rg_k2/(1.-beta2)
        
        lr = self.lr_rate/(torch.sqrt(1+vt)+self.eps)
        
        if lr > 1.0:
            kappa = 1.0
        else:
            kappa = lr
        
        return Gi_k2, Gi_hat, gi_k2, gi_hat, rg_k2, beta1, beta2, kappa, idx
    
    @torch.no_grad()
    def cond_fun(self, 
                 carry:Tuple,
                 )->torch.Tensor:
        
        zi, ui, Gi_k1, Gi_hat, gi_k1, gi_hat, rg_k1, \
            grad_norm, beta1, beta2, kappa, idx = carry

        return (grad_norm>self.tol) & (idx < self.max_iter)
    
    def georce_step(self,
                    carry:Tuple,
                    )->torch.Tensor:
        
        zi, ui, Gi_k1, Gi_hat, gi_k1, gi_hat, rg_k1, \
            grad_norm, beta1, beta2, kappa, idx = carry
        
        Gi_inv = torch.vstack((self.Ginv0, torch.linalg.inv(Gi_hat[1:])))
        zi, ui = self.update_step(zi,
                                  ui,
                                  gi_hat,
                                  Gi_inv,
                                  kappa,
                                  )
        
        gi_k2, Gi_k2, rg_k2 = self.gi(zi,ui)#torch.einsum('tj,tjid,ti->td', un[1:], self.M.DG(xn[1:-1]), un[1:])

        Gi_k2, Gi_hat, gi_k2, gi_hat, rg_k2, beta1, beta2, kappa, idx = self.adaptive_update(Gi_k1, 
                                                                                             Gi_k2, 
                                                                                             gi_k1, 
                                                                                             gi_k2, 
                                                                                             rg_k1, 
                                                                                             rg_k2, 
                                                                                             beta1, 
                                                                                             beta2, 
                                                                                             idx,
                                                                                             )

        grad_norm = torch.linalg.norm(self.Dregenergy(zi, ui, Gi_hat, gi_hat))
        
        return (zi, 
                ui, 
                Gi_k2, 
                Gi_hat, 
                gi_k2, 
                gi_hat, 
                rg_k2, 
                grad_norm, 
                beta1, 
                beta2, 
                kappa, 
                idx+1,
                )
    
    def __call__(self, 
                 z0:torch.Tensor,
                 zN:torch.Tensor,
                 )->torch.Tensor:
        
        shape = z0.shape
        
        self.z0 = z0.detach()
        self.zN = zN.detach()
        self.diff = self.zN-self.z0
        self.dim = len(z0)
        
        self.G0 = self.M.G(z0).detach()
        self.Ginv0 = torch.linalg.inv(self.G0).reshape(1,self.dim,self.dim).detach()
        
        zi, ui = self.initialize()
        
        energy_init = self.energy(zi)
        reg_val_init = torch.sum(torch.linalg.norm(self.score_fun(zi), axis=-1))
        
        if reg_val_init < 1e-6:
            self.lam_norm = self.lam
        else:
            self.lam_norm = self.lam*energy_init/reg_val_init

        gi, Gi, rg = self.gi(zi,ui)#torch.einsum('tj,tjid,ti->td', un[1:], self.M.DG(xn[1:-1]), un[1:])
        grad_norm = torch.linalg.norm(self.Dregenergy(zi, ui, Gi, gi).reshape(-1))
        
        carry = (zi, 
                 ui, 
                 Gi,
                 Gi,
                 gi,
                 gi,
                 rg,
                 grad_norm,
                 self.beta1,
                 self.beta2,
                 self.lr_rate,
                 0,
                 )
        while self.cond_fun(carry):
            carry = self.georce_step(carry)
        zi, ui, Gi_k2, Gi_hat, gi_k2, gi_hat, rg_hat, grad_norm, beta1, beta2, kappa, idx = carry

        zi = torch.vstack((z0, zi, zN))
            
        return zi.reshape(-1,*shape)
    
#%% Prob GEORCE Embedded

class ProbScoreGEORCE_Embedded(ABC):
    def __init__(self,
                 proj_fun:Callable,
                 metric_matrix:Callable,
                 score_fun:Callable,
                 init_fun:Callable=None,
                 lam1:float=1.0,
                 lam2:float=1.0,
                 N:int=100,
                 tol:float=1e-4,
                 max_iter:int=1000,
                 lr_rate:float=0.1,
                 beta1:float=0.5,
                 beta2:float=0.5,
                 eps:float=1e-8,
                 device:str=None,
                 )->None:
        
        self.proj_fun = proj_fun
        self.proj_error = lambda x: torch.sum((proj_fun(x)-x)**2)
        self.metric_matrix = lambda x: metric_matrix(self.proj_fun(x))
        self.score_fun = lambda x: score_fun(self.proj_fun(x))
        
        self.lam1 = lam1
        self.lam2 = lam2
        self.N = N
        self.tol = tol
        self.max_iter = max_iter
        
        self.lr_rate=lr_rate
        self.beta1=beta1
        self.beta2=beta2
        self.eps=eps
        
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        if init_fun is None:
            self.init_fun = lambda z0, zN, N: (zN-z0)*torch.linspace(0.0,
                                                                     1.0,
                                                                     N+1,
                                                                     dtype=z0.dtype,
                                                                     device=self.device)[1:-1].reshape(-1,1)+z0
        else:   
            self.init_fun = init_fun
        
    def __str__(self)->str:
        
        return "Geodesic Computation Object using Control Problem with Probability Flow"
    
    def initialize(self,
                   )->Tuple:
        
        zi = self.init_fun(self.z0,self.zN,self.N)
        zi = self.proj_fun(zi)
        
        val =  torch.vstack((self.z0, zi, self.zN))
        ui = val[1:]-val[:-1]
        
        return zi, ui
    
    @torch.no_grad()
    def energy(self, 
               zi:torch.Tensor,
               )->torch.Tensor:
        
        dz0 = zi[0]-self.z0
        e1 = torch.einsum('i,ij,j->', dz0, self.G0, dz0)
        
        zi = torch.vstack((zi, self.zN))
        Gi = self.metric_matrix(zi[:-1])
        dzi = zi[1:]-zi[:-1]
        
        return e1+torch.sum(torch.einsum('...i,...ij,...j->...', dzi, Gi, dzi))
    
    @torch.no_grad()
    def Dregenergy(self,
                   zi:torch.Tensor,
                   ui:torch.Tensor,
                   Gi:torch.Tensor,
                   gi:torch.Tensor,
                   )->torch.Tensor:
        
        return gi+2.*(torch.einsum('tij,tj->ti', Gi[:-1], ui[:-1])-torch.einsum('tij,tj->ti', Gi[1:], ui[1:]))

    def inner_product(self,
                      zi:torch.Tensor,
                      ui:torch.Tensor,
                      )->torch.Tensor:
        
        Gi = self.metric_matrix(zi)
        proj_val = self.proj_error(zi)
        
        return torch.sum(torch.einsum('...i,...ij,...j->...', ui, Gi, ui)) + self.lam2_norm*proj_val, Gi

    def gi(self,
           zi:torch.Tensor,
           ui:torch.Tensor,
           )->Tuple[torch.Tensor]:
        
        gi, Gi = grad(self.inner_product, has_aux=True)(zi,ui[1:])
        score_val = self.lam1_norm*self.score_fun(zi)
        gi += score_val
        Gi = torch.vstack((self.G0.reshape(-1,self.dim,self.dim),
                           Gi))
        rg = torch.sum(gi**2)
        
        return gi, Gi, rg
    
    @torch.no_grad()
    def update_scheme(self, 
                      gi:torch.Tensor, 
                      Gi_inv:torch.Tensor,
                      )->torch.Tensor:
        
        g_cumsum = torch.flip(torch.cumsum(torch.flip(gi, dims=[0]), dim=0), dims=[0])
        ginv_sum = torch.sum(Gi_inv, axis=0)
        
        rhs = torch.sum(torch.einsum('tij,tj->ti', Gi_inv[:-1], g_cumsum), axis=0)+2.0*self.diff
        
        muN = -torch.linalg.solve(ginv_sum, rhs)
        mui = torch.vstack((muN+g_cumsum, muN))
        
        return mui
    
    @torch.no_grad()
    def update_xi(self,
                  zi:torch.Tensor,
                  alpha:torch.Tensor,
                  ui_hat:torch.Tensor,
                  ui:torch.Tensor,
                  )->torch.Tensor:
        
        return self.z0+torch.cumsum(alpha*ui_hat[:-1]+(1.-alpha)*ui[:-1], axis=0)
    
    @torch.no_grad()
    def update_step(self,
                    zi:torch.Tensor,
                    ui:torch.Tensor,
                    gi_hat:torch.Tensor,
                    Gi_inv:torch.Tensor,
                    kappa:float,
                    )->Tuple[torch.Tensor, torch.Tensor]:
        
        mui = self.update_scheme(gi_hat, Gi_inv)
        ui_hat = -0.5*torch.einsum('tij,tj->ti', Gi_inv, mui)
        zi_hat = self.z0+torch.cumsum(ui_hat[:-1], axis=0)

        return zi+kappa*(zi_hat-zi), ui+kappa*(ui_hat-ui)
    
    @torch.no_grad()
    def adaptive_update(self,
                        Gi_k1:torch.Tensor,
                        Gi_k2:torch.Tensor,
                        gi_k1:torch.Tensor,
                        gi_k2:torch.Tensor,
                        rg_k1:torch.Tensor,
                        rg_k2:torch.Tensor,
                        beta1:torch.Tensor,
                        beta2:torch.Tensor,
                        idx:int,
                        )->Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                 torch.Tensor, torch.Tensor, torch.Tensor]:
    
        Gi_k2 = (1.-self.beta1)*Gi_k2+self.beta1*Gi_k1
        gi_k2 = (1.-self.beta1)*gi_k2+self.beta1*gi_k1
        rg_k2 = (1.-self.beta2)*rg_k2 +self.beta2*rg_k1
        
        beta1 = beta1*self.beta1
        beta2 = beta2*self.beta2

        Gi_hat = Gi_k2/(1.-beta1)
        gi_hat = gi_k2/(1.-beta1)
        vt = rg_k2/(1.-beta2)
        
        lr = self.lr_rate/(torch.sqrt(1+vt)+self.eps)
        if lr > 1.0:
            kappa = 1.0
        else:
            kappa = lr

        return Gi_k2, Gi_hat, gi_k2, gi_hat, rg_k2, beta1, beta2, kappa, idx
    
    @torch.no_grad()
    def cond_fun(self, 
                 carry:Tuple,
                 )->torch.Tensor:
        
        zi, ui, Gi_k1, Gi_hat, gi_k1, gi_hat, rg_k1, \
            grad_norm, beta1, beta2, kappa, idx = carry

        return (grad_norm>self.tol) & (idx < self.max_iter)
    
    def georce_step(self,
                    carry:Tuple,
                    )->torch.Tensor:
        
        zi, ui, Gi_k1, Gi_hat, gi_k1, gi_hat, rg_k1, \
            grad_norm, beta1, beta2, kappa, idx = carry
        
        Gi_inv = torch.vstack((self.Ginv0, torch.linalg.inv(Gi_hat[1:])))
        zi, ui = self.update_step(zi,
                                  ui,
                                  gi_hat,
                                  Gi_inv,
                                  kappa,
                                  )
        
        gi_k2, Gi_k2, rg_k2 = self.gi(zi,ui)#torch.einsum('tj,tjid,ti->td', un[1:], self.M.DG(xn[1:-1]), un[1:])

        Gi_k2, Gi_hat, gi_k2, gi_hat, rg_k2, beta1, beta2, kappa, idx = self.adaptive_update(Gi_k1, 
                                                                                             Gi_k2, 
                                                                                             gi_k1, 
                                                                                             gi_k2, 
                                                                                             rg_k1, 
                                                                                             rg_k2, 
                                                                                             beta1, 
                                                                                             beta2, 
                                                                                             idx,
                                                                                             )

        grad_norm = torch.linalg.norm(self.Dregenergy(zi, ui, Gi_hat, gi_hat))
        
        return (zi, 
                ui, 
                Gi_k2, 
                Gi_hat, 
                gi_k2, 
                gi_hat, 
                rg_k2, 
                grad_norm, 
                beta1, 
                beta2, 
                kappa, 
                idx+1,
                )

    def __call__(self, 
                 z0:torch.Tensor,
                 zN:torch.Tensor,
                 )->torch.Tensor:
        
        shape = z0.shape
        
        self.z0 = z0.detach()
        self.zN = zN.detach()
        self.diff = self.zN-self.z0
        self.dim = len(z0)
        
        self.G0 = self.metric_matrix(z0).detach()
        self.Ginv0 = torch.linalg.inv(self.G0).reshape(1,self.dim,self.dim).detach()
        
        zi, ui = self.initialize()
        
        energy_init = self.energy(zi)
        reg_val_init = torch.sum(torch.linalg.norm(self.score_fun(zi), axis=-1))
        proj_val_init = torch.abs(self.proj_error(zi))
        
        if reg_val_init>1e-6:
            self.lam1_norm = self.lam1*energy_init/reg_val_init
        else:
            self.lam1_norm = self.lam1
            
        if proj_val_init>1e-6:
            self.lam2_norm = self.lam2*energy_init/proj_val_init
        else:
            self.lam2_norm = self.lam2
            
        gi, Gi, rg = self.gi(zi,ui)#torch.einsum('tj,tjid,ti->td', un[1:], self.M.DG(xn[1:-1]), un[1:])
        grad_norm = torch.linalg.norm(self.Dregenergy(zi, ui, Gi, gi).reshape(-1))
        
        carry = (zi, 
                 ui, 
                 Gi,
                 Gi,
                 gi,
                 gi,
                 rg,
                 grad_norm,
                 self.beta1,
                 self.beta2,
                 self.lr_rate,
                 0,
                 )
        
        while self.cond_fun(carry):
            carry = self.georce_step(carry)
        zi, ui, Gi_k2, Gi_hat, gi_k2, gi_hat, rg_hat, grad_norm, beta1, beta2, kappa, idx = carry
        
        zi = torch.vstack((z0, zi, zN))        
        zi = self.proj_fun(zi)
        
        return zi.reshape(-1,*shape)

#%% Probabilistic GEORCE for Euclidean Background Metric

class ProbScoreGEORCE_Euclidean(ABC):
    def __init__(self,
                 score_fun:Callable,
                 init_fun:Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]=None,
                 lam:float=1.0,
                 N:int=100,
                 tol:float=1e-4,
                 max_iter:int=1000,
                 lr_rate:float=0.1,
                 beta1:float=0.5,
                 beta2:float=0.5,
                 eps:float=1e-8,
                 device:str=None,
                 )->None:

        self.score_fun = score_fun
        
        self.lam = lam
        self.N = N
        self.tol = tol
        self.max_iter = max_iter
        
        self.lr_rate=lr_rate
        self.beta1=beta1
        self.beta2=beta2
        self.eps=eps
        
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        if init_fun is None:
            self.init_fun = lambda z0, zN, N: (zN-z0)*torch.linspace(0.0,
                                                                     1.0,
                                                                     N+1,
                                                                     dtype=z0.dtype,
                                                                     device=self.device)[1:-1].reshape(-1,1)+z0
        else:   
            self.init_fun = init_fun
        
    def __str__(self)->str:
        
        return "Geodesic Computation Object using Control Problem with Probability Flow"
    
    def initialize(self,
                   )->Tuple:
        
        zi = self.init_fun(self.z0,self.zN,self.N)
        
        val =  torch.vstack((self.z0, zi, self.zN))
        ui = val[1:]-val[:-1]
        
        return zi, ui
    
    @torch.no_grad()
    def energy(self, 
               zi:torch.Tensor,
               )->torch.Tensor:

        zi = torch.vstack((self.z0, zi, self.zN))
        ui = zi[1:]-zi[:-1]
        
        return torch.sum(ui*ui)

    @torch.no_grad()
    def Dregenergy(self,
                   ui:torch.Tensor,
                   gi:torch.Tensor,
                   )->torch.Tensor:
        
        return gi+2.*(ui[:-1]-ui[1:])

    def gi(self,
           zi:torch.Tensor,
           )->torch.Tensor:
        
        gi = self.lam_norm*self.score_fun(zi)
        
        return gi, torch.sum(gi**2)
    
    @torch.no_grad()
    def update_xi(self,
                  zi:torch.Tensor,
                  alpha:torch.Tensor,
                  ui_hat:torch.Tensor,
                  ui:torch.Tensor,
                  )->torch.Tensor:
        
        return self.z0+torch.cumsum(alpha*ui_hat[:-1]+(1.-alpha)*ui[:-1], axis=0)
    
    @torch.no_grad()
    def update_ui(self,
                  gi:torch.Tensor,
                  )->torch.Tensor:
        
        g_cumsum = torch.vstack((torch.flip(torch.cumsum(torch.flip(gi, dims=[0]), dim=0), dims=[0]), torch.zeros(self.dim, device=gi.device)))
        g_sum = torch.sum(g_cumsum, axis=0)/self.N
        
        return self.diff/self.N+0.5*(g_sum-g_cumsum)
    
    @torch.no_grad()
    def update_step(self,
                    zi:torch.Tensor,
                    ui:torch.Tensor,
                    gi_hat:torch.Tensor,
                    kappa:float,
                    )->Tuple[torch.Tensor, torch.Tensor]:
        
        ui_hat = self.update_ui(gi_hat)
        zi_hat = self.z0+torch.cumsum(ui_hat[:-1], axis=0)

        return zi+kappa*(zi_hat-zi), ui+kappa*(ui_hat-ui)
    
    @torch.no_grad()
    def adaptive_update(self,
                        gi_k1:torch.Tensor,
                        gi_k2:torch.Tensor,
                        rg_k1:torch.Tensor,
                        rg_k2:torch.Tensor,
                        beta1:torch.Tensor,
                        beta2:torch.Tensor,
                        idx:int,
                        )->Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                 torch.Tensor, torch.Tensor, torch.Tensor]:

        gi_k2 = (1.-self.beta1)*gi_k2+self.beta1*gi_k1
        rg_k2 = (1.-self.beta2)*rg_k2 +self.beta2*rg_k1
        
        beta1 = beta1*self.beta1
        beta2 = beta2*self.beta2

        gi_hat = gi_k2/(1.-beta1)
        vt = rg_k2/(1.-beta2)
        
        lr = self.lr_rate/(torch.sqrt(1+vt)+self.eps)
        
        if lr > 1.0:
            kappa = lr
        else:
            kappa = lr

        return gi_k2, gi_hat, rg_k2, beta1, beta2, kappa, idx
    
    @torch.no_grad()
    def cond_fun(self, 
                 carry:Tuple,
                 )->torch.Tensor:
        
        zi, ui, gi_k1, gi_hat, rg_k1, \
            grad_norm, beta1, beta2, kappa, idx = carry

        return (grad_norm>self.tol) & (idx < self.max_iter)
    
    def georce_step(self,
                    carry:Tuple,
                    )->torch.Tensor:
        
        zi, ui, gi_k1, gi_hat, rg_k1, \
            grad_norm, beta1, beta2, kappa, idx = carry

        zi, ui = self.update_step(zi,
                                  ui,
                                  gi_hat,
                                  kappa,
                                  )
        
        gi_k2, rg_k2 = self.gi(zi)#torch.einsum('tj,tjid,ti->td', un[1:], self.M.DG(xn[1:-1]), un[1:])

        gi_k2, gi_hat, rg_k2, beta1, beta2, kappa, idx = self.adaptive_update(gi_k1, 
                                                                              gi_k2, 
                                                                              rg_k1, 
                                                                              rg_k2, 
                                                                              beta1, 
                                                                              beta2, 
                                                                              idx,
                                                                              )

        grad_norm = torch.linalg.norm(self.Dregenergy(ui, gi_hat).reshape(-1))
        
        return (zi, 
                ui, 
                gi_k2, 
                gi_hat, 
                rg_k2, 
                grad_norm, 
                beta1, 
                beta2, 
                kappa, 
                idx+1,
                )
    
    def __call__(self, 
                 z0:torch.Tensor,
                 zN:torch.Tensor,
                 )->torch.Tensor:
        
        shape = z0.shape
        
        self.z0 = z0.reshape(-1).detach()
        self.zN = zN.reshape(-1).detach()
        self.diff = self.zN-self.z0
        self.dim = len(z0)
        
        zi, ui = self.initialize()
        
        energy_init = self.energy(zi)
        reg_val_init = torch.sum(torch.linalg.norm(self.score_fun(zi), axis=-1))
        
        if reg_val_init>1e-6:
            self.lam_norm = self.lam*energy_init/reg_val_init
        else:
            self.lam_norm = self.lam
        
        gi, rg = self.gi(zi)#torch.einsum('tj,tjid,ti->td', un[1:], self.M.DG(xn[1:-1]), un[1:])
        grad_norm = torch.linalg.norm(self.Dregenergy(ui, gi).reshape(-1))
        
        carry = (zi, 
                 ui, 
                 gi,
                 gi,
                 rg,
                 grad_norm,
                 self.beta1,
                 self.beta2,
                 self.lr_rate,
                 0,
                 )
        while self.cond_fun(carry):
            carry = self.georce_step(carry)
        zi, ui, gi_k2, gi_hat, rg_hat, grad_norm, beta1, beta2, kappa, idx = carry
        zi = torch.vstack((z0, zi, zN))        
        
        return zi.reshape(-1,*shape)