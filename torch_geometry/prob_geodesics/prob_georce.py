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

#%% Gradient Descent Estimation of Geodesics

class ProbGEORCE(ABC):
    def __init__(self,
                 M:RiemannianManifold,
                 reg_fun:Callable,
                 init_fun:Callable=None,
                 lam:float=1.0,
                 N:int=100,
                 tol:float=1e-4,
                 max_iter:int=1000,
                 line_search_params:Dict = {'rho': 0.5},
                 device:str=None,
                 )->None:
        
        self.M = M
        self.reg_fun = reg_fun
        
        self.lam = lam
        self.N = N
        self.tol = tol
        self.max_iter = max_iter
        self.line_search_params = line_search_params
        
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
    def reg_energy(self, 
                   zi:torch.Tensor,
                   *args,
                   )->torch.Tensor:
        
        energy = self.energy(zi)
        reg_val = self.reg_fun(zi)
        
        return energy + self.lam_norm*reg_val
    
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
        reg_val = self.reg_fun(zi)
        
        return torch.sum(torch.einsum('...i,...ij,...j->...', ui, Gi, ui))+self.lam_norm*reg_val, Gi

    def gi(self,
           zi:torch.Tensor,
           ui:torch.Tensor,
           )->Tuple[torch.Tensor]:
        
        gi, Gi = grad(self.inner_product, has_aux=True)(zi,ui[1:])
        Gi = torch.vstack((self.G0.reshape(-1,self.M.dim,self.M.dim),
                           Gi))
        
        return gi, Gi
    
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
        
        return (self.z0+torch.cumsum(alpha*ui_hat[:-1]+(1.-alpha)*ui[:-1], axis=0),)
    
    @torch.no_grad()
    def cond_fun(self, 
                 carry:Tuple,
                 )->torch.Tensor:
        
        zi, ui, Gi, gi, Gi_inv, grad_val, idx = carry
        
        grad_norm = torch.linalg.norm(grad_val)

        return (grad_norm>self.tol) & (idx < self.max_iter)
    
    def georce_step(self,
                    carry:Tuple,
                    )->torch.Tensor:
        
        zi, ui, Gi, gi, Gi_inv, grad_val, idx = carry
        
        mui = self.update_scheme(gi, Gi_inv)
        
        ui_hat = -0.5*torch.einsum('tij,tj->ti', Gi_inv, mui)
        tau = self.line_search((zi,), grad_val, ui_hat, ui)

        ui = tau*ui_hat+(1.-tau)*ui
        zi = self.z0+torch.cumsum(ui[:-1], axis=0)
        
        gi, Gi = self.gi(zi,ui)#torch.einsum('tj,tjid,ti->td', un[1:], self.M.DG(xn[1:-1]), un[1:])
        Gi_inv = torch.vstack((self.Ginv0, torch.linalg.inv(Gi[1:])))
        grad_val = self.Dregenergy(zi, ui, Gi, gi)
        
        return (zi, ui, Gi, gi, Gi_inv, grad_val, idx+1)
    
    def __call__(self, 
                 z0:torch.Tensor,
                 zN:torch.Tensor,
                 )->torch.Tensor:
        
        shape = z0.shape
        
        self.line_search = Backtracking(obj_fun=self.reg_energy,
                                        update_fun=self.update_xi,
                                        **self.line_search_params,
                                        )
        
        self.z0 = z0.detach()
        self.zN = zN.detach()
        self.diff = self.zN-self.z0
        self.dim = len(z0)
        
        self.G0 = self.M.G(z0).detach()
        self.Ginv0 = torch.linalg.inv(self.G0).reshape(1,self.dim,self.dim).detach()
        
        zi, ui = self.initialize()

        energy_init = self.energy(zi)
        reg_val_init = torch.abs(self.reg_fun(zi))
        
        if reg_val_init < 1e-6:
            self.lam_norm = self.lam
        else:
            self.lam_norm = self.lam*energy_init/reg_val_init
        
        gi, Gi = self.gi(zi,ui)#torch.einsum('tj,tjid,ti->td', un[1:], self.M.DG(xn[1:-1]), un[1:])
        Gi_inv = torch.vstack((self.Ginv0, torch.linalg.inv(Gi[1:])))
        grad_val = self.Dregenergy(zi, ui, Gi, gi)
        
        carry=(zi, 
               ui, 
               Gi, 
               gi, 
               Gi_inv, 
               grad_val, 
               0,
               )
        
        while self.cond_fun(carry):
            carry = self.georce_step(carry)
        zi, ui, Gi, gi, gi_inv, grad_val, idx = carry
            
        zi = torch.vstack((z0, zi, zN))
            
        return zi.reshape(-1,*shape)
    
#%% Prob GEORCE Embedded

class ProbGEORCE_Embedded(ABC):
    def __init__(self,
                 proj_fun:Callable,
                 metric_matrix:Callable,
                 reg_fun:Callable,
                 init_fun:Callable=None,
                 lam1:float=1.0,
                 lam2:float=1.0,
                 N:int=100,
                 tol:float=1e-4,
                 max_iter:int=1000,
                 line_search_params:Dict = {'rho': 0.5},
                 device:str=None,
                 )->None:
        
        self.proj_fun = proj_fun
        self.proj_error = lambda x: torch.sum((proj_fun(x)-x)**2)
        self.metric_matrix = lambda x: metric_matrix(self.proj_fun(x))
        self.reg_fun = lambda x: reg_fun(self.proj_fun(x))
        
        self.lam1 = lam1
        self.lam2 = lam2
        self.N = N
        self.tol = tol
        self.max_iter = max_iter
        self.line_search_params = line_search_params
        
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
    def reg_energy(self, 
                   zi:torch.Tensor,
                   *args,
                   )->torch.Tensor:
        
        energy = self.energy(zi)
        reg_val = self.reg_fun(zi)
        proj_val = self.proj_error(zi)
        
        return energy + self.lam1_norm*reg_val + self.lam2_norm*proj_val
    
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
        reg_val = self.reg_fun(zi)
        proj_val = self.proj_error(zi)
        
        return torch.sum(torch.einsum('...i,...ij,...j->...', ui, Gi, ui))+self.lam1_norm*reg_val \
            +self.lam2_norm*proj_val, Gi

    def gi(self,
           zi:torch.Tensor,
           ui:torch.Tensor,
           )->Tuple[torch.Tensor]:
        
        gi, Gi = grad(self.inner_product, has_aux=True)(zi,ui[1:])
        Gi = torch.vstack((self.G0.reshape(-1,self.dim,self.dim),
                           Gi))
        
        return gi, Gi
    
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
        
        return (self.z0+torch.cumsum(alpha*ui_hat[:-1]+(1.-alpha)*ui[:-1], axis=0),)
    
    def cond_fun(self, 
                 carry:Tuple,
                 )->torch.Tensor:
        
        zi, ui, Gi, gi, Gi_inv, grad_val, idx = carry
        
        grad_norm = torch.linalg.norm(grad_val)

        return (grad_norm>self.tol) & (idx < self.max_iter)
    
    def georce_step(self,
                    carry:Tuple,
                    )->torch.Tensor:
        
        zi, ui, Gi, gi, Gi_inv, grad_val, idx = carry
        
        mui = self.update_scheme(gi, Gi_inv)
        
        ui_hat = -0.5*torch.einsum('tij,tj->ti', Gi_inv, mui)
        tau = self.line_search((zi,), grad_val, ui_hat, ui)

        ui = tau*ui_hat+(1.-tau)*ui
        zi = self.z0+torch.cumsum(ui[:-1], axis=0)
        
        gi, Gi = self.gi(zi,ui)#torch.einsum('tj,tjid,ti->td', un[1:], self.M.DG(xn[1:-1]), un[1:])
        Gi_inv = torch.vstack((self.Ginv0, torch.linalg.inv(Gi[1:])))
        grad_val = self.Dregenergy(zi, ui, Gi, gi)
        
        return (zi, ui, Gi, gi, Gi_inv, grad_val, idx+1)

    def __call__(self, 
                 z0:torch.Tensor,
                 zN:torch.Tensor,
                 )->torch.Tensor:
        
        shape = z0.shape
        
        self.line_search = Backtracking(obj_fun=self.reg_energy,
                                        update_fun=self.update_xi,
                                        **self.line_search_params,
                                        )
        
        self.z0 = z0.detach()
        self.zN = zN.detach()
        self.diff = self.zN-self.z0
        self.dim = len(z0)
        
        self.G0 = self.metric_matrix(z0).detach()
        self.Ginv0 = torch.linalg.inv(self.G0).reshape(1,self.dim,self.dim).detach()
        
        zi, ui = self.initialize()

        energy_init = self.energy(zi)
        reg_val_init = torch.abs(self.reg_fun(zi))
        proj_val_init = torch.abs(self.proj_error(zi))
        
        if reg_val_init>1e-6:
            self.lam1_norm = self.lam1*energy_init/reg_val_init
        else:
            self.lam1_norm = self.lam1
            
        if proj_val_init>1e-6:
            self.lam2_norm = self.lam2*energy_init/proj_val_init
        else:
            self.lam2_norm = self.lam2
        
        gi, Gi = self.gi(zi,ui)#torch.einsum('tj,tjid,ti->td', un[1:], self.M.DG(xn[1:-1]), un[1:])
        Gi_inv = torch.vstack((self.Ginv0, torch.linalg.inv(Gi[1:])))
        grad_val = self.Dregenergy(zi, ui, Gi, gi)
        
        carry = (zi,
                 ui,
                 Gi,
                 gi,
                 Gi_inv,
                 grad_val,
                 0,
                 )
        while self.cond_fun(carry):
            carry = self.georce_step(carry)
        
        zi, ui, Gi, gi, gi_inv, grad_val, idx = carry
        zi = torch.vstack((z0, zi, zN))
        
        zi = self.proj_fun(zi)
        
        return zi.reshape(-1,*shape)

#%% Probabilistic GEORCE for Euclidean Background Metric

class ProbGEORCE_Euclidean(ABC):
    def __init__(self,
                 reg_fun:Callable,
                 init_fun:Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]=None,
                 lam:float=1.0,
                 N:int=100,
                 tol:float=1e-4,
                 max_iter:int=1000,
                 line_search_params:Dict = {'rho': 0.5},
                 device:str=None,
                 )->None:

        self.reg_fun = reg_fun
        
        self.lam = lam
        self.N = N
        self.tol = tol
        self.max_iter = max_iter
        self.line_search_params = line_search_params
        
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
    def reg_energy(self, 
                   zi:torch.Tensor,
                   *args,
                   )->torch.Tensor:

        reg_val = self.reg_fun(zi)
        energy = self.energy(zi)
        
        return energy+self.lam_norm*reg_val

    @torch.no_grad()
    def Dregenergy(self,
                   ui:torch.Tensor,
                   gi:torch.Tensor,
                   )->torch.Tensor:
        
        return gi+2.*(ui[:-1]-ui[1:])

    def inner_product(self,
                      zi:torch.Tensor,
                      )->torch.Tensor:
        
        return self.lam_norm*self.reg_fun(zi)
    
    def gi(self,
           zi:torch.Tensor,
           )->torch.Tensor:
        
        return grad(self.inner_product)(zi)
    
    @torch.no_grad()
    def update_xi(self,
                  zi:torch.Tensor,
                  alpha:torch.Tensor,
                  ui_hat:torch.Tensor,
                  ui:torch.Tensor,
                  )->torch.Tensor:
        
        return (self.z0+torch.cumsum(alpha*ui_hat[:-1]+(1.-alpha)*ui[:-1], axis=0),)
    
    @torch.no_grad()
    def update_ui(self,
                  gi:torch.Tensor,
                  )->torch.Tensor:
        
        g_cumsum = torch.vstack((torch.flip(torch.cumsum(torch.flip(gi, dims=[0]), dim=0), dims=[0]), torch.zeros(self.dim, device=gi.device)))
        g_sum = torch.sum(g_cumsum, axis=0)/self.N
        
        return self.diff/self.N+0.5*(g_sum-g_cumsum)
    
    @torch.no_grad()
    def cond_fun(self, 
                 carry:Tuple,
                 )->torch.Tensor:
        
        zi, ui, gi, grad_val, idx = carry
        
        grad_norm = torch.linalg.norm(grad_val)

        return (grad_norm>self.tol) & (idx < self.max_iter)
    
    def georce_step(self,
                    carry:Tuple,
                    )->torch.Tensor:
        
        zi, ui, gi, grad_val, idx = carry

        ui_hat = self.update_ui(gi)
        tau = self.line_search((zi,), grad_val, ui_hat, ui)

        ui = tau*ui_hat+(1.-tau)*ui
        zi = self.z0+torch.cumsum(ui[:-1], axis=0)
        
        gi = self.gi(zi)#torch.einsum('tj,tjid,ti->td', un[1:], self.M.DG(xn[1:-1]), un[1:])
        grad_val = self.Dregenergy(ui, gi)
        
        return (zi, ui, gi, grad_val, idx+1)
    
    def __call__(self, 
                 z0:torch.Tensor,
                 zN:torch.Tensor,
                 )->torch.Tensor:
        
        shape = z0.shape
        
        self.line_search = Backtracking(obj_fun=self.reg_energy,
                                        update_fun=self.update_xi,
                                        **self.line_search_params,
                                        )
        
        self.z0 = z0.reshape(-1).detach()
        self.zN = zN.reshape(-1).detach()
        self.diff = self.zN-self.z0
        self.dim = len(z0)
        
        zi, ui = self.initialize()

        energy_init = self.energy(zi)
        reg_val_init = torch.abs(self.reg_fun(zi))
        
        if reg_val_init>1e-6:
            self.lam_norm = self.lam*energy_init/reg_val_init
        else:
            self.lam_norm = self.lam
        
        gi = self.gi(zi)#torch.einsum('tj,tjid,ti->td', un[1:], self.M.DG(xn[1:-1]), un[1:])
        grad_val = self.Dregenergy(ui, gi)
        
        carry = zi, ui, gi, grad_val, 0
        while self.cond_fun(carry):
            carry = self.georce_step(carry)
        zi, ui, gi, grad_val, idx = carry
        
        zi = torch.vstack((z0, zi, zN))
            
        return zi.reshape(-1,*shape)
    
#%% ProbGEORCE NoiseDiffusion

class ProbGEORCE_NoiseDiffusion(ABC):
    """Probabilistic GEORCE with NoiseDiffusion

    Estimates geodesics for the Euclidean metric under the constaint that
    these are within the probability distribituon using the score function.

    Attributes:
        reg_fun: regularizing function that is vectorized and returns a scalar
        init_fun: initilization function for the initial curve
        lam: lambda that determines the deviation between the geodesic and probaility flow
        N: number of grid points, in total the outputet curve will have (N+1) grid points
        tol: the tolerance for convergence
        max_iter: the maximum number of iterations
        line_search_params: the parameters for the line search (max_iter and rho), where rho is backtracking parameter
    """
    def __init__(self,
                 interpolater:Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 alpha:Callable=lambda s: torch.cos(0.5*torch.pi*s),
                 beta:Callable=lambda s: torch.sin(0.5*torch.pi*s),
                 mu:Callable|None= lambda s: None,
                 nu:Callable|None= lambda s: None,
                 gamma:float=0.0,
                 sigma:float=1.0,
                 boundary:float=2.0,
                 device:str=None,
                 )->None:
        """Initializes the instance of ProbGEORCE with Euclidean background metric.

        Args:
          reg_fun: regularizing function that is vectorized
          init_fun: initilization function for the initial curve
          lam: lambda that determines the deviation between the geodesic and probaility flow
          N: number of grid points, in total the outputet curve will have (N+1) grid points
          tol: the tolerance for convergence
          max_iter: the maximum number of iterations
          line_search_params: the parameters for the line search (max_iter and rho), where rho is backtracking parameter
        """
        
        self.interpolater = interpolater
        
        self.alpha = alpha
        self.beta = beta

        self.mu = mu if mu is None else lambda s: 1.2*self.alpha(s)/(self.alpha(s)+self.beta(s))
        self.nu = nu if nu is None else lambda s: 1.2*self.beta(s)/(self.alpha(s)+self.beta(s))
        
        self.gamma = gamma
        self.sigma = sigma
        self.boundary = boundary
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
    def __str__(self)->str:
        
        return "NoiseDiffusion with different ProbGEORCE interpolation"

    def __call__(self,
                 z0:Tensor,
                 zN:Tensor,
                 x0:Tensor,
                 xN:Tensor,
                 )->Tensor:
        
        shape = z0.shape
        
        z0 = z0.reshape(-1)
        zN = zN.reshape(-1)
        x0 = x0.reshape(-1)
        xN = xN.reshape(-1)
        
        s = torch.linspace(0,1,self.interpolater.N+1,
                           device=self.device,
                           )[1:-1].reshape(-1,1)

        #alpha=math.cos(math.radians(s*90))
        #beta=math.sin(math.radians(s*90))
        alpha = torch.cos(0.5*torch.pi*s)
        beta = torch.sin(0.5*torch.pi*s)
        
        mu = vmap(self.mu)(s)
        nu = vmap(self.nu)(s)
        eps = self.sigma*torch.randn_like(z0)
        
        l=alpha/beta
        
        alpha=((1-self.gamma*self.gamma)*l*l/(l*l+1))**0.5
        beta=((1-self.gamma*self.gamma)/(l*l+1))**0.5
        
        noise_curve = self.interpolater(z0,zN)[1:-1]
        data_curve = self.interpolater(x0, xN)[1:-1]
        
        noise_latent = noise_curve - data_curve + \
            (mu*x0 + nu * xN)+self.gamma*eps

        curve=torch.clip(noise_latent,-self.boundary,self.boundary)
        curve = torch.vstack((z0, curve, zN))
        
        return curve.reshape(-1, *shape)