#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 12:01:26 2024

@author: fmry
"""

#%% Sources

#%% Modules

from jax_geometry.setup import *

from jax_geometry.manifolds import RiemannianManifold

#%% Gradient Descent Estimation of Geodesics

class JAXOptimization(ABC):
    def __init__(self,
                 M:RiemannianManifold,
                 reg_fun:Callable,
                 init_fun:Callable=None,
                 lam:float=1.0,
                 lr_rate:float=1.0,
                 optimizer:Callable=None,
                 N:int=100,
                 tol:float=1e-4,
                 max_iter:int=1000,
                 )->None:
        
        self.M = M
        self.reg_fun = reg_fun
        
        self.lam = lam
        self.N = N
        self.tol = tol
        self.max_iter = max_iter
        
        if init_fun is None:
            self.init_fun = lambda z0, zN, N: (zN-z0)*jnp.linspace(0.0,
                                                                   1.0,
                                                                   N,
                                                                   endpoint=False,
                                                                   dtype=z0.dtype)[1:].reshape(-1,1)+z0
        else:
            self.init_fun = init_fun
            
        if optimizer is None:
            self.opt_init, self.opt_update, self.get_params = optimizers.adam(lr_rate)
        else:
            self.opt_init, self.opt_update, self.get_params = optimizer(lr_rate)
        
        if init_fun is None:
            self.init_fun = lambda z0, zT, T: (zT-z0)*jnp.linspace(0.0,
                                                                   1.0,
                                                                   T,
                                                                   endpoint=False,
                                                                   dtype=z0.dtype)[1:].reshape(-1,1)+z0
        
    def __str__(self)->str:
        
        return "Geodesic Computation Object using Control Problem with Probability Flow"
    
    def initialize(self,
                   )->Tuple:
        
        zi = self.init_fun(self.z0,self.zN,self.N)
        
        val =  jnp.vstack((self.z0, zi, self.zN))
        ui = val[1:]-val[:-1]
        
        return zi, ui
    
    def energy(self, 
               zi:Array,
               )->Array:
        
        dz0 = zi[0]-self.z0
        e1 = jnp.einsum('i,ij,j->', dz0, self.G0, dz0)
        
        zi = jnp.vstack((zi, self.zN))
        Gi = vmap(self.M.G)(lax.stop_gradient(zi[:-1]))
        dzi = zi[1:]-zi[:-1]
        
        return e1+jnp.sum(jnp.einsum('...i,...ij,...j->...', dzi, Gi, dzi))
    
    def reg_energy(self, 
                   zi:Array,
                   *args,
                   )->Array:
        
        energy = self.energy(zi)
        reg_val = self.reg_fun(zi)
        
        return energy + self.lam_norm*reg_val
    
    def Denergy(self,
                 zt:Array,
                 )->Array:
         
         return grad(lambda z: self.reg_energy(z))(zt)
    
    def cond_fun(self, 
                 carry:Tuple[Array, Array, Array, int],
                 )->Array:
        
        zt, grad, opt_state, idx = carry

        norm_grad = jnp.linalg.norm(grad.reshape(-1))

        return (norm_grad>self.tol) & (idx < self.max_iter)
    
    def step_fun(self,
                 carry:Tuple[Array, Array, Array, int],
                 )->Array:
        
        zt, grad, opt_state, idx = carry
        
        opt_state = self.opt_update(idx, grad, opt_state)
        zt = self.get_params(opt_state)

        grad = self.Denergy(zt)
        
        return (zt, grad, opt_state, idx+1)
    
    def __call__(self, 
                 z0:Array,
                 zN:Array,
                 )->Array:
        
        shape = z0.shape
        
        self.z0 = lax.stop_gradient(z0)
        self.zN = lax.stop_gradient(zN)
        self.diff = self.zN-self.z0
        self.dim = len(z0)
        
        self.G0 = lax.stop_gradient(self.M.G(z0))
        
        zi, _ = self.initialize()
        
        opt_state = self.opt_init(zi)

        energy_init = lax.stop_gradient(self.energy(zi))
        reg_val_init = lax.stop_gradient(self.reg_fun(zi))
        
        
        self.lam_norm = lax.cond(jnp.abs(reg_val_init) < 1e-4,
                                 lambda *_: self.lam,
                                 lambda *_: self.lam*energy_init/reg_val_init,
                                 )
        
        grad_val = self.Denergy(zi)
    
        zi, grad, _, idx = lax.while_loop(self.cond_fun, 
                                          self.step_fun,
                                          init_val=(zi, grad_val, opt_state, 0)
                                          )

        reg_energy = self.reg_energy(zi)
        zi = jnp.vstack((z0, zi, zN))
            
        return zi.reshape(-1,*shape)