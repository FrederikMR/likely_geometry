#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 09:54:30 2024

@author: fmry
"""

#%% Sources

#%% Modules

import torch
from torch import vmap
from torchdiffeq import odeint
from torch.func import jacfwd, grad

from typing import Callable, Tuple

from abc import ABC

#%% Riemannian Manifold

class RiemannianManifold(ABC):
    def __init__(self,
                 G:Callable[[torch.Tensor], torch.Tensor]=None,
                 f:Callable[[torch.Tensor], torch.Tensor]=None,
                 invf:Callable[[torch.Tensor],torch.Tensor]=None,
                 )->None:
        
        self.f = f
        self.invf = invf
        if ((G is None) and (f is None)):
            raise ValueError("Both the metric, g, and chart, f, is not defined")
        elif (G is None):
            self.G = lambda z: self.pull_back_metric(z)
        else:
            self.G = G
            
        return
        
    def __str__(self)->str:
        
        return "Riemannian Manifold base object"
    
    
    def Jf(self, z: torch.Tensor) -> torch.Tensor:
        if self.f is None:
            raise ValueError("Both the pull-back map is not defined")
    
        # J_single: (d,) -> (m, d)
        J_single = jacfwd(self.f)
    
        # Make it batchable: apply vmap once per batch dimension
        f = J_single
        for _ in range(z.ndim - 1):
            f = vmap(f)
    
        # Now f : (..., d) -> (..., m, d)
        return f(z)
        
    def pull_back_metric(self,
                         z:torch.Tensor
                         )->torch.Tensor:
        
        if self.f is None:
            raise ValueError("Both the pull-back map is not defined")
        else:
            Jf = self.Jf(z)
            return torch.einsum('...ik,...il->...kl', Jf, Jf)
    
    def DG(self, z: torch.Tensor) -> torch.Tensor:
        """
        Computes the Jacobian of G at z.
        Supports any number of batch dimensions.
    
        G : (..., d) -> (..., m)
        DG: (..., d) -> (..., m, d)
        """
    
        # G must be defined
        if self.G is None:
            raise ValueError("The map G is not defined")
    
        # Single-sample Jacobian: (d,) -> (m, d)
        J_single = jacfwd(self.G, randomness='same')
    
        # Make it batchable by wrapping in vmap
        J = J_single
        for _ in range(z.ndim - 1):
            J = vmap(J)
    
        # Apply to batched input
        return J(z)
    
    def inner_product(self,
                      z:torch.Tensor,
                      u:torch.Tensor,
                      )->torch.Tensor:
        
        G = self.G(z)
        
        return torch.einsum('...i,...ij,...j->...', u,G,u)
    
    def Ginv(self,
             z:torch.Tensor
             )->torch.Tensor:
        
        return torch.linalg.inv(self.G(z))
    
    def christoffel_symbols(self,
                            z:torch.Tensor
                            )->torch.Tensor:
        
        Dgx = self.DG(z)
        gsharpx = self.Ginv(z)
        
        return 0.5*(torch.einsum('...im,...kml->...ikl',gsharpx,Dgx)
                   +torch.einsum('...im,...lmk->...ikl',gsharpx,Dgx)
                   -torch.einsum('...im,...klm->...ikl',gsharpx,Dgx))
    
    def geodesic_equation(self,
                          z:torch.Tensor,
                          v:torch.Tensor
                          )->torch.Tensor:
        
        Gamma = self.christoffel_symbols(z)

        dx1t = v
        dx2t = -torch.einsum('...ikl,...k,...l->...i',Gamma,v,v)
        
        return torch.hstack((dx1t,dx2t))
    
    def Exp(self,
            z:torch.Tensor,
            v:torch.Tensor,
            T:int=100,
            )->torch.Tensor:
        
        def dif_fun(t,y):
            
            z = y[:self.dim]
            v = y[self.dim:]
            
            return self.geodesic_equation(z, v)
        
        dim = len(z)
        
        t_grid = torch.linspace(0., 1., T) 
        zs = odeint(dif_fun, 
                    torch.hstack((z, v)), 
                    t_grid, 
                    method='rk4',
                    )

        zs = zs[:,:dim]
        
        return zs
    
    def Exp_ode(self,
                z:torch.Tensor,
                v:torch.Tensor,
                T:int=100,
                )->torch.Tensor:
        
        def dif_fun(t,y):
            
            z = y[:self.dim]
            v = y[self.dim:]
            
            return self.geodesic_equation(z, v)
        
        dim = len(z)
        
        t_grid = torch.linspace(0., 1., T) 
        zs = odeint(dif_fun, 
                    torch.hstack((z, v)), 
                    t_grid, 
                    method='rk4',
                    )

        zs = zs[:,:dim]
        
        return zs

    def energy(self, 
               gamma:torch.Tensor,
               )->torch.Tensor:
        
        T = len(gamma)-1
        dt = 1.0/T
        dgamma = (gamma[1:]-gamma[:-1])*T
        
        g =self.G(gamma)
        integrand = torch.einsum('...ti,...tij,...tj->...t', dgamma, g[:-1], dgamma)
        
        dt = len(integrand)*dt
        
        return torch.trapezoid(integrand, dt)
    
    def length(self,
               gamma:torch.Tensor,
               )->torch.Tensor:
        
        T = len(gamma)-1
        dt = 1.0/T
        
        dgamma = (gamma[1:]-gamma[:-1])*T
        
        g = self.G(gamma)
        integrand = torch.sqrt(torch.einsum('...ti,...tij,...tj->...t', dgamma, g[:-1], dgamma))
        
        dt = len(integrand)*dt
            
        return torch.trapezoid(integrand, dt)
    
    def length_frechet(self, 
                       zt:torch.Tensor,
                       z_obs:torch.Tensor,
                       z_mu:torch.Tensor,
                       )->torch.Tensor:
        
        def step_length(length:torch.Tensor,
                        y:Tuple,
                        )->Tuple:
            
            z0, z_path = y
            
            length += self.path_length_frechet(z0, z_path, z_mu, G0)**2
            
            return (length,)*2
        
        G0 = self.G(z_mu)
        
        length = 0.0
        for z0,z_path in zip(z_obs, zt):
            length = step_length(length, (z0,z_path))
        
        return length
    
    def path_length_frechet(self, 
                            zT:torch.Tensor,
                            zt:torch.Tensor,
                            mu:torch.Tensor,
                            G0:torch.Tensor,
                            )->torch.Tensor:
        
        term1 = zt[0]-mu
        val1 = torch.sqrt(torch.einsum('...i,...ij,...j->', term1, G0, term1))
        
        term2 = zt[1:]-zt[:-1]
        Gt = self.G(zt)
        val2 = torch.sqrt(torch.einsum('...ti,...tij,...tj->...t', term2, Gt[:-1], term2))
        
        term3 = zT-zt[-1]
        val3 = torch.sqrt(torch.einsum('...i,...ij,...j->...', term3, Gt[-1], term3))
        
        return val1+torch.sum(val2)+val3
    
    def indicatrix(self,
                   z:torch.Tensor,
                   N_points:int=100,
                   *args,
                   )->torch.Tensor:
        
        theta = torch.linspace(0.,2*torch.pi,N_points)
        u = torch.vstack((torch.cos(theta), torch.sin(theta))).T
        
        norm = torch.vmap(self.inner_product, in_axes=(None, 0))(z,u)
        
        return torch.einsum('ij,i->ij', u, 1./norm)
    
#%% Lambda Manifold

class LambdaManifold(ABC):
    def __init__(self,
                 M:RiemannianManifold,
                 S:Callable[[torch.Tensor], torch.Tensor],
                 gradS:Callable[[torch.Tensor], torch.Tensor]=None,
                 lam:float=1.0,
                 )->None:
        
        self.M = M
        self.S = S
        self.lam = lam
            
        if gradS is None:
            self.gradS = self.gradS_numeric
        else:
            self.gradS = gradS
        
        return
        
    def __str__(self)->str:
        
        return "Lambda Manifold base object"
    
    def inner_product(self,
                      z:torch.Tensor,
                      u:torch.Tensor,
                      )->torch.Tensor:
        
        G = self.M.G(z)
        
        return torch.einsum('...i,...ij,...j->...', u,G,u) + self.lam*self.S(z)
    
    def Ginv(self,
             z:torch.Tensor
             )->torch.Tensor:
        
        return torch.linalg.inv(self.M.G(z))
    
    def christoffel_symbols(self,
                            z:torch.Tensor
                            )->torch.Tensor:
        
        return self.M.christoffel_symbols(z)
    
    
    def gradS_numeric(self, z: torch.Tensor) -> torch.Tensor:
        """
        Computes grad S(z) for z of shape (..., d),
        returning (..., d).
        """
        # gradient function for a single (d,) input
        g_single = grad(self.S)
    
        # make batchable by wrapping in vmap once per batch dimension
        g = g_single
        for _ in range(z.ndim - 1):
            g = vmap(g)
    
        return g(z)
        
    def geodesic_equation(self,
                          z:torch.Tensor,
                          v:torch.Tensor
                          )->torch.Tensor:
        
        Gamma = self.M.christoffel_symbols(z)
        grad_s = self.gradS(z)
        Ginv = self.M.Ginv(z)

        dx1t = v
        dx2t = (
            -torch.einsum('...ikl,...k,...l->...i',Gamma,v,v) - 0.5*self.lam*torch.einsum('...ij,...j->...i', Ginv, grad_s)
            )
        
        return torch.hstack((dx1t,dx2t))
    
    def Exp(self,
            z:torch.Tensor,
            v:torch.Tensor,
            T:int=100,
            )->torch.Tensor:
        
        def dif_fun(t,y):
            
            z = y[:self.M.dim]
            v = y[self.M.dim:]
            
            return self.geodesic_equation(z, v)
        
        dim = len(z)
        
        t_grid = torch.linspace(0., 1., T) 
        zs = odeint(dif_fun, 
                    torch.hstack((z, v)), 
                    t_grid, 
                    method='rk4',
                    )
        
        zs = zs[:,:dim]
        
        return zs
    
    def Exp_ode(self,
                z:torch.Tensor,
                v:torch.Tensor,
                T:int=100,
                )->torch.Tensor:
        
        def dif_fun(t,y):
            
            z = y[:self.M.dim]
            v = y[self.M.dim:]
            
            return self.geodesic_equation(z, v)
        
        dim = len(z)
        
        t_grid = torch.linspace(0., 1., T, device=z.device) 
        zs = odeint(dif_fun, 
                    torch.hstack((z, v)), 
                    t_grid, 
                    method='rk4',
                    )
        
        zs = zs[:,:dim]
        
        return zs

    def energy(self, 
               gamma:torch.Tensor,
               )->torch.Tensor:
        
        T = len(gamma)-1
        dt = 1.0/T
        dgamma = (gamma[1:]-gamma[:-1])*T
        
        g = self.M.G(gamma)
        integrand = torch.einsum('...ti,...tij,...tj->...t', dgamma, g[:-1], dgamma)
        integrand += self.lam*self.S(gamma)
        
        return torch.trapezoid(integrand, dx=dt)
    
    def length(self,
               gamma:torch.Tensor,
               )->torch.Tensor:
        
        T = len(gamma)-1
        dt = 1.0/T
        
        dgamma = (gamma[1:]-gamma[:-1])*T
        
        g = self.M.G(gamma)
        integrand = torch.sqrt(torch.einsum('ti,tij,tj->t', dgamma, g[:-1], dgamma) + self.lam*self.S(gamma))
            
        return torch.trapezoid(integrand, dx=dt)
    
    def length_frechet(self, 
                       zt:torch.Tensor,
                       z_obs:torch.Tensor,
                       z_mu:torch.Tensor,
                       )->torch.Tensor:
        
        def step_length(length:torch.Tensor,
                         y:Tuple,
                         )->Tuple:
            
            z0, z_path = y
            
            length += self.path_length_frechet(z0, z_path, z_mu, G0)**2
            
            return (length,)*2
        
        G0 = self.M.G(z_mu)
        length = 0.0
        for z0,z_path in zip(z_obs,zt):
            length = step_length(length, (z0,z_path))
        
        return length
    
    def path_length_frechet(self, 
                            zT:torch.Tensor,
                            zt:torch.Tensor,
                            mu:torch.Tensor,
                            G0:torch.Tensor,
                            )->torch.Tensor:
        
        term1 = zt[0]-mu
        val1 = torch.sqrt(torch.einsum('i,ij,j->', term1, G0, term1) + self.lam*self.S(mu))
        
        term2 = zt[1:]-zt[:-1]
        Gt = self.M.G(zt)
        S_val = self.S(zt)
        val2 = torch.sqrt(torch.einsum('ti,tij,tj->t', term2, Gt[:-1], term2) + self.lam*S_val[:-1])
        
        term3 = zT-zt[-1]
        val3 = torch.sqrt(torch.einsum('i,ij,j->', term3, Gt[-1], term3) + self.lam*S_val[-1])
        
        return val1+torch.sum(val2)+val3
    
    def indicatrix(self,
                   z:torch.Tensor,
                   N_points:int=100,
                   *args,
                   )->torch.Tensor:
        
        theta = torch.linspace(0.,2*torch.pi,N_points)
        u = torch.vstack((torch.cos(theta), torch.sin(theta))).T
        
        norm = torch.vmap(self.inner_product, in_axes=(None, 0))(z,u)
        
        return torch.einsum('ij,i->ij', u, 1./norm)