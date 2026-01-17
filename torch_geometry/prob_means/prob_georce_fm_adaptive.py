#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 17:06:07 2025

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

#%% Gradient Descent Estimation of Geodesics

class ProbGEORCEFM_Adaptive(ABC):
    def __init__(self,
                 M:RiemannianManifold,
                 reg_fun:Callable[[torch.Tensor],torch.Tensor],
                 init_fun:Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]=None,
                 lam:float=1.0,
                 N_grid:int=100,
                 tol:float=1e-4,
                 max_iter:int=1000,
                 lr_rate:float=0.1,
                 beta1:float=0.5,
                 beta2:float=0.5,
                 eps:float=1e-8,
                 device:str=None,
                 )->None:
        
        self.M = M
        self.reg_fun = reg_fun
        self.lam = lam
        self.N_grid = N_grid
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
        
        return "Geodesic Computation Object using Control Problem"
    
    def init_curve(self, 
                   z0:torch.Tensor, 
                   zT:torch.Tensor,
                   )->torch.Tensor:
        
        zi = self.init_fun(z0, zT, self.N_grid)
        total = torch.vstack((z0, zi, zT))
        ui = total[1:]-total[:-1]
        
        return zi, ui
    
    @torch.no_grad()
    def reg_energy(self,
                   zi:torch.Tensor,
                   z_mu:torch.Tensor,
                   *args,
                   )->torch.Tensor:
        
        energy = self.energy(zi, z_mu)
        reg_val = self.wi*(self.reg_fun(zi) + self.N_data*self.reg_fun(z_mu))
        
        return torch.sum(self.wi*energy) + self.lam_norm*torch.sum(reg_val)
    
    @torch.no_grad()
    def energy(self,
               zi:torch.Tensor,
               z_mu:torch.Tensor,
               *args,
               )->torch.Tensor:
        
        term1 = zi[:,0]-self.z_obs
        val1 = torch.einsum('...i,...ij,...j->...', term1, self.G0, term1)
        
        term2 = zi[:,1:]-zi[:,:-1]
        Gi = self.M.G(zi)
        val2 = torch.einsum('...i,...ij,...j->...', term2, Gi[:,:-1], term2)
        
        term3 = z_mu-zi[:,-1]
        val3 = torch.einsum('...i,...ij,...j->...', term3, Gi[:,-1], term3)
        
        energy = val1+torch.sum(val2, axis=1)+val3
        
        return torch.sum(self.wi*energy)
        
    @torch.no_grad()
    def Dregenergy_frechet(self,
                           zi:torch.Tensor,
                           ui:torch.Tensor,
                           z_mu:torch.Tensor,
                           Gi:torch.Tensor,
                           gi:torch.Tensor,
                           )->torch.Tensor:

        Gi = torch.concatenate((self.G0.reshape(self.N_data, -1, self.dim, self.dim), 
                                Gi,
                                ),
                               axis=1)
        
        dcurve = torch.mean(gi+2.*(torch.einsum('...ij,...j->...i', Gi[:,:-1], ui[:,:-1])-\
                            torch.einsum('...ij,...j->...i', Gi[:,1:], ui[:,1:])), axis=0)
        dmu = 2.*torch.mean(torch.einsum('...ij,...i->...j', Gi[:,-1], ui[:,-1]), axis=0)
        
        return torch.hstack((dcurve.reshape(-1), dmu))
    
    def inner_product(self,
                      zi:torch.Tensor,
                      ui:torch.Tensor,
                      )->torch.Tensor:
            
        Gi = self.M.G(zi)
        reg_val = torch.sum(self.reg_fun(zi))

        return torch.sum(torch.einsum('...i,...ij,...j->...', ui, Gi, ui))+self.lam_norm*reg_val, Gi
    
    def gi(self,
           zi:torch.Tensor,
           ui:torch.Tensor,
           )->torch.Tensor:
        
        gi, Gi = grad(self.inner_product, has_aux=True)(zi, ui)
        
        return gi, Gi, torch.mean(torch.sum(gi**2, axis=(1,2)), axis=0)
        
    @torch.no_grad()
    def curve_update(self, 
                     z_mu:torch.Tensor,
                     g_cumsum:torch.Tensor, 
                     Gi_inv:torch.Tensor,
                     ginv_sum_inv:torch.Tensor,
                     )->torch.Tensor:
        
        diff = torch.einsum('...,...i->...i', self.wi, z_mu-self.z_obs)
        
        rhs = torch.sum(torch.einsum('...ij,...j->...i', Gi_inv[:,:-1], g_cumsum), axis=1)+2.0*diff

        muT = -torch.einsum('...ij,...j->...i', ginv_sum_inv, rhs).reshape(-1,1,self.dim)
        mui = torch.concatenate((muT+g_cumsum, muT), axis=1)
        
        return mui
    
    @torch.no_grad()
    def frechet_update(self,
                       g_cumsum:torch.Tensor,
                       Gi_inv:torch.Tensor,
                       ginv_sum_inv:torch.Tensor,
                       )->torch.Tensor:
        
        rhs = torch.einsum('k,kji,ki->kj', self.wi, ginv_sum_inv, self.z_obs) \
            -0.5*torch.einsum('kji, ki->kj', ginv_sum_inv,
                            torch.sum(torch.einsum('ktij,ktj->kti', Gi_inv[:,:-1], g_cumsum), axis=1),
                            )
            
        lhs = torch.einsum('t,tij->tij', self.wi, ginv_sum_inv)

        mu = torch.linalg.solve(torch.sum(lhs, axis=0), 
                                torch.sum(rhs, axis=0),
                                )
        
        return mu
    
    @torch.no_grad()
    def update_step(self,
                    zi:torch.Tensor,
                    ui:torch.Tensor,
                    z_mu:torch.Tensor,
                    gi_hat:torch.Tensor,
                    Gi_inv:torch.Tensor,
                    kappa:float,
                    )->Tuple[torch.Tensor, torch.Tensor]:
        
        g_cumsum = torch.cumsum(gi_hat.flip(dims=[1]), dim=1).flip(dims=[1])
        ginv_sum_inv = torch.linalg.inv(torch.sum(Gi_inv, axis=1))
        
        z_mu_hat = self.frechet_update(g_cumsum, Gi_inv, ginv_sum_inv)
        mui = self.curve_update(z_mu_hat, g_cumsum, Gi_inv, ginv_sum_inv)
        ui_hat = -0.5*torch.einsum('k,ktij,ktj->kti', 1./self.wi, Gi_inv, mui)

        zi_hat = self.z_obs.reshape(-1,1,self.dim)+torch.cumsum(ui_hat[:,:-1], axis=1)

        return zi+kappa*(zi_hat-zi), ui+kappa*(ui_hat-ui), z_mu + kappa*(z_mu_hat-z_mu)
    
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
                 carry:Tuple[torch.Tensor,torch.Tensor,torch.Tensor, torch.Tensor, int],
                 )->torch.Tensor:
        
        zi, ui, z_mu, Gi_k1, Gi_hat, gi_k1, gi_hat, rg_k1, \
            grad_norm, beta1, beta2, kappa, idx = carry

        return (grad_norm>self.tol) & (idx < self.max_iter)
    
    def while_step(self,
                     carry:Tuple[torch.Tensor,torch.Tensor,torch.Tensor, torch.Tensor, int],
                     )->torch.Tensor:
        
        zi, ui, z_mu, Gi_k1, Gi_hat, gi_k1, gi_hat, rg_k1, \
            grad_norm, beta1, beta2, kappa, idx = carry
            
            
        Gi_inv = torch.concatenate((self.Ginv0.reshape(self.N_data, -1, self.dim, self.dim), 
                                  torch.linalg.inv(Gi_hat),
                                  ),
                                 axis=1)
        zi, ui, z_mu = self.update_step(zi, 
                                        ui, 
                                        z_mu, 
                                        gi_hat,
                                        Gi_inv,
                                        kappa,
                                        )
        gi_k2, Gi_k2, rg_k2 = self.gi(zi,ui[:,1:])
        
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
        
        grad_norm = torch.linalg.norm(self.Dregenergy_frechet(zi, ui, z_mu, Gi_hat, gi_hat).reshape(-1))
        
        return (zi, 
                ui, 
                z_mu,
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
                 z_obs:torch.Tensor,
                 wi:torch.Tensor=None,
                 z_mu_init:torch.Tensor=None,
                 )->torch.Tensor:
        
        self.z_obs = z_obs
        self.N_data, self.dim = self.z_obs.shape
        
        self.G0 = self.M.G(self.z_obs).detach()
        self.Ginv0 = torch.linalg.inv(self.G0).detach()
        
        if wi is None:
            self.wi = torch.ones(self.N_data, device=self.device)
        else:
            self.wi = wi
        
        if z_mu_init is None:
            z_mu_init = torch.mean(self.z_obs, axis=0)

        zi, ui = vmap(self.init_curve, in_dims=(0,None))(self.z_obs, z_mu_init)
        
        energy_init = self.energy(zi, z_mu_init)
        reg_val_init = torch.abs(torch.sum(self.reg_fun(zi)) + self.reg_fun(z_mu_init))
        
        if reg_val_init < 1e-6:
            self.lam_norm = self.lam
        else:
            self.lam_norm = self.lam*energy_init/reg_val_init

        gi, Gi, rg = self.gi(zi, ui[:,1:])
        grad_norm = torch.linalg.norm(self.Dregenergy_frechet(zi, ui, z_mu_init, Gi, gi).reshape(-1))
        
        carry = (zi,
                 ui,
                 z_mu_init,
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
            carry = self.while_step(carry)
        zi, ui, z_mu, Gi_k1, Gi_hat, gi_k1, gi_hat, rg_k1, \
            grad_norm, beta1, beta2, kappa, idx = carry
                
        zi = zi.flip(dims=[1])
            
        # reshape to match (10,1,2) and (1,1,2)
        end = self.z_obs.unsqueeze(1)          # (10, 1, 2)
        start   = z_mu.view(1, 1, self.dim).expand(self.N_data, 1, self.dim)  # (10, 1, 2)
        
        # concatenate along the middle dimension
        zi = torch.cat([start, zi, end], dim=1)
            
        return z_mu, zi
    
#%% Gradient Descent Estimation of Geodesics

class ProbGEORCEFM_Embedded_Adaptive(ABC):
    def __init__(self,
                 proj_fun:Callable,
                 metric_matrix:Callable,
                 reg_fun:Callable[[torch.Tensor],torch.Tensor],
                 init_fun:Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]=None,
                 lam1:float=1.0,
                 lam2:float=1.0,
                 N_grid:int=100,
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
        self.reg_fun = lambda x: reg_fun(self.proj_fun(x))
        
        self.lam1 = lam1
        self.lam2 = lam2
        self.N_grid = N_grid
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
        
        return "Geodesic Computation Object using Control Problem"
    
    def init_curve(self, 
                   z0:torch.Tensor, 
                   zT:torch.Tensor,
                   )->torch.Tensor:
        
        zi = self.init_fun(z0, zT, self.N_grid)
        total = torch.vstack((z0, zi, zT))
        ui = total[1:]-total[:-1]
        
        return zi, ui
    
    @torch.no_grad()
    def reg_energy(self,
                   zi:torch.Tensor,
                   z_mu:torch.Tensor,
                   *args,
                   )->torch.Tensor:
        
        energy = self.energy(zi, z_mu)
        reg_val = self.wi*(self.reg_fun(zi) + self.N_data*self.reg_fun(z_mu))
        proj_val = self.wi*(self.proj_error(zi) + self.N_data*self.proj_error(z_mu))
        
        return torch.sum(self.wi*energy) + self.lam1_norm*torch.sum(reg_val) + + self.lam2_norm*torch.sum(proj_val)
    
    @torch.no_grad()
    def energy(self,
               zi:torch.Tensor,
               z_mu:torch.Tensor,
               *args,
               )->torch.Tensor:
        
        term1 = zi[:,0]-self.z_obs
        val1 = torch.einsum('...i,...ij,...j->...', term1, self.G0, term1)
        
        term2 = zi[:,1:]-zi[:,:-1]
        Gi = self.metric_matrix(zi)
        val2 = torch.einsum('...i,...ij,...j->...', term2, Gi[:,:-1], term2)
        
        term3 = z_mu-zi[:,-1]
        val3 = torch.einsum('...i,...ij,...j->...', term3, Gi[:,-1], term3)
        
        energy = val1+torch.sum(val2, axis=1)+val3
        
        return torch.sum(self.wi*energy)
        
    @torch.no_grad()
    def Dregenergy_frechet(self,
                           zi:torch.Tensor,
                           ui:torch.Tensor,
                           z_mu:torch.Tensor,
                           Gi:torch.Tensor,
                           gi:torch.Tensor,
                           )->torch.Tensor:

        Gi = torch.concatenate((self.G0.reshape(self.N_data, -1, self.dim, self.dim), 
                                Gi,
                                ),
                               axis=1)
        
        dcurve = torch.mean(gi+2.*(torch.einsum('...ij,...j->...i', Gi[:,:-1], ui[:,:-1])-\
                            torch.einsum('...ij,...j->...i', Gi[:,1:], ui[:,1:])), axis=0)
        dmu = 2.*torch.mean(torch.einsum('...ij,...i->...j', Gi[:,-1], ui[:,-1]), axis=0)
        
        return torch.hstack((dcurve.reshape(-1), dmu))
    
    def inner_product(self,
                      zi:torch.Tensor,
                      ui:torch.Tensor,
                      )->torch.Tensor:
            
        Gi = self.metric_matrix(zi)
        reg_val = torch.sum(self.reg_fun(zi))
        proj_val = torch.sum(self.proj_error(zi))

        return torch.sum(torch.einsum('...i,...ij,...j->...', ui, Gi, ui))+self.lam1_norm*reg_val + self.lam2_norm*proj_val, Gi
    
    def gi(self,
           zi:torch.Tensor,
           ui:torch.Tensor,
           )->torch.Tensor:
        
        gi, Gi = grad(self.inner_product, has_aux=True)(zi, ui)
        
        return gi, Gi, torch.mean(torch.sum(gi**2, axis=(1,2)), axis=0)
        
    @torch.no_grad()
    def curve_update(self, 
                     z_mu:torch.Tensor,
                     g_cumsum:torch.Tensor, 
                     Gi_inv:torch.Tensor,
                     ginv_sum_inv:torch.Tensor,
                     )->torch.Tensor:
        
        diff = torch.einsum('...,...i->...i', self.wi, z_mu-self.z_obs)
        
        rhs = torch.sum(torch.einsum('...ij,...j->...i', Gi_inv[:,:-1], g_cumsum), axis=1)+2.0*diff

        muT = -torch.einsum('...ij,...j->...i', ginv_sum_inv, rhs).reshape(-1,1,self.dim)
        mui = torch.concatenate((muT+g_cumsum, muT), axis=1)
        
        return mui
    
    @torch.no_grad()
    def frechet_update(self,
                       g_cumsum:torch.Tensor,
                       Gi_inv:torch.Tensor,
                       ginv_sum_inv:torch.Tensor,
                       )->torch.Tensor:
        
        rhs = torch.einsum('k,kji,ki->kj', self.wi, ginv_sum_inv, self.z_obs) \
            -0.5*torch.einsum('kji, ki->kj', ginv_sum_inv,
                            torch.sum(torch.einsum('ktij,ktj->kti', Gi_inv[:,:-1], g_cumsum), axis=1),
                            )
            
        lhs = torch.einsum('t,tij->tij', self.wi, ginv_sum_inv)

        mu = torch.linalg.solve(torch.sum(lhs, axis=0), 
                                torch.sum(rhs, axis=0),
                                )
        
        return mu
    
    @torch.no_grad()
    def update_step(self,
                    zi:torch.Tensor,
                    ui:torch.Tensor,
                    z_mu:torch.Tensor,
                    gi_hat:torch.Tensor,
                    Gi_inv:torch.Tensor,
                    kappa:float,
                    )->Tuple[torch.Tensor, torch.Tensor]:
        
        g_cumsum = torch.cumsum(gi_hat.flip(dims=[1]), dim=1).flip(dims=[1])
        ginv_sum_inv = torch.linalg.inv(torch.sum(Gi_inv, axis=1))
        
        z_mu_hat = self.frechet_update(g_cumsum, Gi_inv, ginv_sum_inv)
        mui = self.curve_update(z_mu_hat, g_cumsum, Gi_inv, ginv_sum_inv)
        ui_hat = -0.5*torch.einsum('k,ktij,ktj->kti', 1./self.wi, Gi_inv, mui)

        zi_hat = self.z_obs.reshape(-1,1,self.dim)+torch.cumsum(ui_hat[:,:-1], axis=1)

        return zi+kappa*(zi_hat-zi), ui+kappa*(ui_hat-ui), z_mu + kappa*(z_mu_hat-z_mu)
    
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
                 carry:Tuple[torch.Tensor,torch.Tensor,torch.Tensor, torch.Tensor, int],
                 )->torch.Tensor:
        
        zi, ui, z_mu, Gi_k1, Gi_hat, gi_k1, gi_hat, rg_k1, \
            grad_norm, beta1, beta2, kappa, idx = carry

        return (grad_norm>self.tol) & (idx < self.max_iter)
    
    def while_step(self,
                     carry:Tuple[torch.Tensor,torch.Tensor,torch.Tensor, torch.Tensor, int],
                     )->torch.Tensor:
        
        zi, ui, z_mu, Gi_k1, Gi_hat, gi_k1, gi_hat, rg_k1, \
            grad_norm, beta1, beta2, kappa, idx = carry
            
        Gi_inv = torch.concatenate((self.Ginv0.reshape(self.N_data, -1, self.dim, self.dim), 
                                    torch.linalg.inv(Gi_hat),
                                    ),
                                   axis=1)
        zi, ui, z_mu = self.update_step(zi, 
                                        ui, 
                                        z_mu, 
                                        gi_hat,
                                        Gi_inv,
                                        kappa,
                                        )
        gi_k2, Gi_k2, rg_k2 = self.gi(zi, ui[:,1:])
        
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
        
        grad_norm = torch.linalg.norm(self.Dregenergy_frechet(zi, ui, z_mu, Gi_hat, gi_hat).reshape(-1))
        
        return (zi, 
                ui, 
                z_mu,
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
                 z_obs:torch.Tensor,
                 wi:torch.Tensor=None,
                 z_mu_init:torch.Tensor=None,
                 )->torch.Tensor:

        self.z_obs = z_obs
        self.N_data, self.dim = self.z_obs.shape
        
        self.G0 = self.metric_matrix(self.z_obs).detach()
        self.Ginv0 = torch.linalg.inv(self.G0).detach()
        
        if wi is None:
            self.wi = torch.ones(self.N_data, device=self.device)
        else:
            self.wi = wi
        
        if z_mu_init is None:
            z_mu_init = torch.mean(self.z_obs, axis=0)

        zi, ui = vmap(self.init_curve, in_dims=(0,None))(self.z_obs, z_mu_init)
        
        energy_init = self.energy(zi, z_mu_init)
        reg_val_init = torch.abs(torch.sum(self.reg_fun(zi)) + self.reg_fun(z_mu_init))
        proj_val_init = torch.abs(torch.sum(self.proj_error(zi) + self.proj_error(z_mu_init)))
        
        if reg_val_init>1e-6:
            self.lam1_norm = self.lam1*energy_init/reg_val_init
        else:
            self.lam1_norm = self.lam1
            
        if proj_val_init>1e-6:
            self.lam2_norm = self.lam2*energy_init/proj_val_init
        else:
            self.lam2_norm = self.lam2

        gi, Gi, rg = self.gi(zi, ui[:,1:])
        grad_norm = torch.linalg.norm(self.Dregenergy_frechet(zi, ui, z_mu_init, Gi, gi).reshape(-1))
        
        carry = (zi,
                 ui,
                 z_mu_init,
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
            carry = self.while_step(carry)
        zi, ui, z_mu, Gi_k1, Gi_hat, gi_k1, gi_hat, rg_k1, \
            grad_norm, beta1, beta2, kappa, idx = carry
        
        zi = zi.flip(dims=[1])
            
        # reshape to match (10,1,2) and (1,1,2)
        end = self.z_obs.unsqueeze(1)          # (10, 1, 2)
        start   = z_mu.view(1, 1, self.dim).expand(self.N_data, 1, self.dim)  # (10, 1, 2)
        
        # concatenate along the middle dimension
        zi = torch.cat([start, zi, end], dim=1)
            
        return z_mu, zi
    
#%% Gradient Descent Estimation of Geodesics

class ProbGEORCEFM_Euclidean_Adaptive(ABC):
    def __init__(self,
                 reg_fun:Callable[[torch.Tensor],torch.Tensor],
                 init_fun:Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]=None,
                 lam:float=1.0,
                 N_grid:int=100,
                 tol:float=1e-4,
                 max_iter:int=1000,
                 lr_rate:float=0.1,
                 beta1:float=0.5,
                 beta2:float=0.5,
                 eps:float=1e-8,
                 device:str=None,
                 )->None:

        self.reg_fun = reg_fun
        self.lam = lam
        self.N_grid = N_grid
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
        
        return "Geodesic Computation Object using Control Problem"
    
    def init_curve(self, 
                   z0:torch.Tensor, 
                   zT:torch.Tensor,
                   )->torch.Tensor:
        
        zi = self.init_fun(z0, zT, self.N_grid)
        total = torch.vstack((z0, zi, zT))
        ui = total[1:]-total[:-1]
        
        return zi, ui
    
    @torch.no_grad()
    def reg_energy(self,
                   zi:torch.Tensor,
                   z_mu:torch.Tensor,
                   *args,
                   )->torch.Tensor:
        
        energy = self.energy(zi, z_mu)
        reg_val = self.wi*(self.reg_fun(zi) + self.N_data*self.reg_fun(z_mu))
        
        return torch.sum(self.wi*energy) + self.lam_norm*torch.sum(reg_val)
    
    @torch.no_grad()
    def energy(self,
               zi:torch.Tensor,
               z_mu:torch.Tensor,
               *args,
               )->torch.Tensor:
        
        term1 = zi[:,0]-self.z_obs
        val1 = torch.einsum('...i,...i->...', term1, term1)
        
        term2 = zi[:,1:]-zi[:,:-1]
        val2 = torch.einsum('...i,...i->...', term2, term2)
        
        term3 = z_mu-zi[:,-1]
        val3 = torch.einsum('...i,...i->...', term3, term3)
        
        energy = val1+torch.sum(val2, axis=1)+val3
        
        return torch.sum(self.wi*energy)
        
    @torch.no_grad()
    def Dregenergy_frechet(self,
                           zi:torch.Tensor,
                           ui:torch.Tensor,
                           z_mu:torch.Tensor,
                           gi:torch.Tensor,
                           )->torch.Tensor:
        
        dcurve = torch.mean(gi+2.*(ui[:,:-1]-ui[:,1:]), axis=0)
        dmu = 2.*torch.mean(ui[:,-1], axis=0)
        
        return torch.hstack((dcurve.reshape(-1), dmu))
    
    def inner_product(self,
                      zi:torch.Tensor,
                      )->torch.Tensor:

        reg_val = torch.sum(self.reg_fun(zi))

        return self.lam_norm*reg_val
    
    def gi(self,
           zi:torch.Tensor,
           )->torch.Tensor:
        
        gi = grad(self.inner_product)(zi)
        
        return gi, torch.mean(torch.sum(gi**2, axis=(1,2)), axis=0)
        
    @torch.no_grad()
    def curve_update(self, 
                     z_mu:torch.Tensor,
                     g_cumsum:torch.Tensor, 
                     )->torch.Tensor:

        g_cumsum = torch.cat(
            (g_cumsum, torch.zeros(self.N_data, 1, self.dim, device=g_cumsum.device, dtype=g_cumsum.dtype)),
            dim=1
            )
        term1 = 0.5*(torch.sum(g_cumsum, axis=1)[:,None,:]/self.N_grid-g_cumsum)/self.wi[:,None,None]
        term2 = (z_mu - self.z_obs)/self.N_grid
        
        return term1 + term2[:,None,:]
    
    @torch.no_grad()
    def frechet_update(self,
                       g_cumsum:torch.Tensor,
                       )->torch.Tensor:

        sum_w = torch.sum(self.wi)
        term1 = torch.sum(self.wi[:,None]*self.z_obs, axis=0) - 0.5*torch.sum(g_cumsum, axis=(0,1))
        
        return term1/sum_w
    
    @torch.no_grad()
    def update_step(self,
                    zi:torch.Tensor,
                    ui:torch.Tensor,
                    z_mu:torch.Tensor,
                    gi_hat:torch.Tensor,
                    kappa:float,
                    )->Tuple[torch.Tensor, torch.Tensor]:
        
        g_cumsum = torch.cumsum(gi_hat.flip(1), dim=1).flip(1)
        
        z_mu_hat = self.frechet_update(g_cumsum)
        ui_hat = self.curve_update(z_mu_hat, g_cumsum)

        zi_hat = self.z_obs.reshape(-1,1,self.dim)+torch.cumsum(ui_hat[:,:-1], axis=1)

        return zi+kappa*(zi_hat-zi), ui+kappa*(ui_hat-ui), z_mu + kappa*(z_mu_hat-z_mu)
    
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
            kappa = 1.0
        else:
            kappa = lr
        
        return gi_k2, gi_hat, rg_k2, beta1, beta2, kappa, idx

    @torch.no_grad()
    def cond_fun(self, 
                 carry:Tuple[torch.Tensor,torch.Tensor,torch.Tensor, torch.Tensor, int],
                 )->torch.Tensor:
        
        zi, ui, z_mu, gi_k1, gi_hat, rg_k1, \
            grad_norm, beta1, beta2, kappa, idx = carry

        return (grad_norm>self.tol) & (idx < self.max_iter)
    
    def while_step(self,
                     carry:Tuple[torch.Tensor,torch.Tensor,torch.Tensor, torch.Tensor, int],
                     )->torch.Tensor:
        
        zi, ui, z_mu, gi_k1, gi_hat, rg_k1, \
            grad_norm, beta1, beta2, kappa, idx = carry

        zi, ui, z_mu = self.update_step(zi, 
                                        ui, 
                                        z_mu, 
                                        gi_hat,
                                        kappa,
                                        )
        gi_k2, rg_k2 = self.gi(zi)
        
        gi_k2, gi_hat, rg_k2, beta1, beta2, kappa, idx = self.adaptive_update(gi_k1, 
                                                                              gi_k2, 
                                                                              rg_k1, 
                                                                              rg_k2, 
                                                                              beta1, 
                                                                              beta2, 
                                                                              idx,
                                                                              )
        
        grad_norm = torch.linalg.norm(self.Dregenergy_frechet(zi, ui, z_mu, gi_hat).reshape(-1))
        
        return (zi, 
                ui, 
                z_mu,
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
                 z_obs:torch.Tensor,
                 wi:torch.Tensor=None,
                 z_mu_init:torch.Tensor=None,
                 )->torch.Tensor:
        
        self.z_obs = z_obs
        self.N_data, self.dim = self.z_obs.shape
        
        if wi is None:
            self.wi = torch.ones(self.N_data, device=self.device)
        else:
            self.wi = wi
        
        if z_mu_init is None:
            z_mu_init = torch.mean(self.z_obs, axis=0)

        zi, ui = vmap(self.init_curve, in_dims=(0,None))(self.z_obs, z_mu_init)
        
        energy_init = self.energy(zi, z_mu_init)
        reg_val_init = torch.abs(torch.sum(self.reg_fun(zi)) + self.reg_fun(z_mu_init))
        
        if reg_val_init < 1e-6:
            self.lam_norm = self.lam
        else:
            self.lam_norm = self.lam*energy_init/reg_val_init

        gi, rg = self.gi(zi)
        grad_norm = torch.linalg.norm(self.Dregenergy_frechet(zi, ui, z_mu_init, gi).reshape(-1))
        
        carry = (zi,
                 ui,
                 z_mu_init,
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
            carry = self.while_step(carry)
        zi, ui, z_mu, gi_k1, gi_hat, rg_k1, \
            grad_norm, beta1, beta2, kappa, idx = carry
            
        zi = zi.flip(dims=[1])
            
        # reshape to match (10,1,2) and (1,1,2)
        end = self.z_obs.unsqueeze(1)          # (10, 1, 2)
        start   = z_mu.view(1, 1, self.dim).expand(self.N_data, 1, self.dim)  # (10, 1, 2)
        
        # concatenate along the middle dimension
        zi = torch.cat([start, zi, end], dim=1)
            
        return z_mu, zi