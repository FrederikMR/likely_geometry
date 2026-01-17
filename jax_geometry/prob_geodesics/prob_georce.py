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
from jax_geometry.line_search import Backtracking

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
                 )->None:
        
        self.M = M
        self.reg_fun = reg_fun
        
        self.lam = lam
        self.N = N
        self.tol = tol
        self.max_iter = max_iter
        self.line_search_params = line_search_params
        
        if init_fun is None:
            self.init_fun = lambda z0, zN, N: (zN-z0)*jnp.linspace(0.0,
                                                                   1.0,
                                                                   N,
                                                                   endpoint=False,
                                                                   dtype=z0.dtype)[1:].reshape(-1,1)+z0
        else:
            self.init_fun = init_fun
        
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
        
        return lax.stop_gradient(energy + self.lam_norm*reg_val)
    
    def Dregenergy(self,
                   zi:Array,
                   ui:Array,
                   Gi:Array,
                   gi:Array,
                   )->Array:
        
        return gi+2.*(jnp.einsum('tij,tj->ti', Gi[:-1], ui[:-1])-jnp.einsum('tij,tj->ti', Gi[1:], ui[1:]))

    def inner_product(self,
                      zi:Array,
                      ui:Array,
                      )->Array:
        
        Gi = vmap(self.M.G)(zi)
        reg_val = self.reg_fun(zi)
        
        return jnp.sum(jnp.einsum('...i,...ij,...j->...', ui, Gi, ui))+self.lam_norm*reg_val, Gi

    def gi(self,
           zi:Array,
           ui:Array,
           )->Tuple[Array]:
        
        gi, Gi = lax.stop_gradient(grad(self.inner_product, has_aux=True)(zi,ui[1:]))
        Gi = jnp.vstack((self.G0.reshape(-1,self.M.dim,self.M.dim),
                         Gi))
        
        return gi, Gi
    
    def update_scheme(self, 
                      gi:Array, 
                      Gi_inv:Array,
                      )->Array:
        
        g_cumsum = jnp.cumsum(gi[::-1], axis=0)[::-1]
        ginv_sum = jnp.sum(Gi_inv, axis=0)
        
        rhs = jnp.sum(jnp.einsum('tij,tj->ti', Gi_inv[:-1], g_cumsum), axis=0)+2.0*self.diff
        
        muN = -jnp.linalg.solve(ginv_sum, rhs)
        mui = jnp.vstack((muN+g_cumsum, muN))
        
        return mui
    
    def update_xi(self,
                  zi:Array,
                  alpha:Array,
                  ui_hat:Array,
                  ui:Array,
                  )->Array:
        
        return (self.z0+jnp.cumsum(alpha*ui_hat[:-1]+(1.-alpha)*ui[:-1], axis=0),)
    
    def cond_fun(self, 
                 carry:Tuple,
                 )->Array:
        
        zi, ui, Gi, gi, Gi_inv, grad_val, idx = carry
        
        grad_norm = jnp.linalg.norm(grad_val)

        return (grad_norm>self.tol) & (idx < self.max_iter)
    
    def georce_step(self,
                    carry:Tuple,
                    )->Array:
        
        zi, ui, Gi, gi, Gi_inv, grad_val, idx = carry
        
        mui = self.update_scheme(gi, Gi_inv)
        
        ui_hat = -0.5*jnp.einsum('tij,tj->ti', Gi_inv, mui)
        tau = self.line_search((zi,), grad_val, ui_hat, ui)

        ui = tau*ui_hat+(1.-tau)*ui
        zi = self.z0+jnp.cumsum(ui[:-1], axis=0)
        
        gi, Gi = self.gi(zi,ui)#jnp.einsum('tj,tjid,ti->td', un[1:], self.M.DG(xn[1:-1]), un[1:])
        Gi_inv = jnp.vstack((self.Ginv0, jnp.linalg.inv(Gi[1:])))
        grad_val = self.Dregenergy(zi, ui, Gi, gi)
        
        return (zi, ui, Gi, gi, Gi_inv, grad_val, idx+1)
    
    def __call__(self, 
                 z0:Array,
                 zN:Array,
                 )->Array:
        
        shape = z0.shape
        
        self.line_search = Backtracking(obj_fun=self.reg_energy,
                                        update_fun=self.update_xi,
                                        **self.line_search_params,
                                        )
        
        self.z0 = lax.stop_gradient(z0)
        self.zN = lax.stop_gradient(zN)
        self.diff = self.zN-self.z0
        self.dim = len(z0)
        
        self.G0 = lax.stop_gradient(self.M.G(z0))
        self.Ginv0 = jnp.linalg.inv(self.G0).reshape(1,self.dim,self.dim)
        
        zi, ui = self.initialize()

        energy_init = lax.stop_gradient(self.energy(zi))
        reg_val_init = jnp.abs(lax.stop_gradient(self.reg_fun(zi)))
        
        self.lam_norm = lax.cond(reg_val_init < 1e-6,
                                 lambda *_: self.lam,
                                 lambda *_: self.lam*energy_init/reg_val_init,
                                 )
        
        gi, Gi = self.gi(zi,ui)#jnp.einsum('tj,tjid,ti->td', un[1:], self.M.DG(xn[1:-1]), un[1:])
        Gi_inv = jnp.vstack((self.Ginv0, jnp.linalg.inv(Gi[1:])))
        grad_val = self.Dregenergy(zi, ui, Gi, gi)
        
        zi, ui, Gi, gi, Gi_inv, grad_val, idx = lax.while_loop(self.cond_fun, 
                                                               self.georce_step, 
                                                               init_val=(zi, 
                                                                         ui, 
                                                                         Gi, 
                                                                         gi, 
                                                                         Gi_inv, 
                                                                         grad_val, 
                                                                         0),
                                                               )
        zi = jnp.vstack((z0, zi, zN))
            
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
                 )->None:
        
        self.proj_fun = proj_fun
        self.proj_error = lambda x: jnp.sum((proj_fun(x)-x)**2)
        self.metric_matrix = lambda x: metric_matrix(self.proj_fun(x))
        self.reg_fun = lambda x: reg_fun(self.proj_fun(x))
        
        self.lam1 = lam1
        self.lam2 = lam2
        self.N = N
        self.tol = tol
        self.max_iter = max_iter
        self.line_search_params = line_search_params
        
        if init_fun is None:
            self.init_fun = lambda z0, zT, T: (zT-z0)*jnp.linspace(0.0,
                                                                   1.0,
                                                                   T,
                                                                   endpoint=False,
                                                                   dtype=z0.dtype)[1:].reshape(-1,1)+z0
        else:
            self.init_fun = init_fun
        
    def __str__(self)->str:
        
        return "Geodesic Computation Object using Control Problem with Probability Flow"
    
    def initialize(self,
                   )->Tuple:
        
        zi = self.init_fun(self.z0,self.zN,self.N)
        zi = self.proj_fun(zi)
        
        val =  jnp.vstack((self.z0, zi, self.zN))
        ui = val[1:]-val[:-1]
        
        return zi, ui
    
    def energy(self, 
               zi:Array,
               )->Array:
        
        dz0 = zi[0]-self.z0
        e1 = jnp.einsum('i,ij,j->', dz0, self.G0, dz0)
        
        zi = jnp.vstack((zi, self.zN))
        Gi = self.metric_matrix(lax.stop_gradient(zi[:-1]))
        dzi = zi[1:]-zi[:-1]
        
        return e1+jnp.sum(jnp.einsum('...i,...ij,...j->...', dzi, Gi, dzi))
    
    def reg_energy(self, 
                   zi:Array,
                   *args,
                   )->Array:
        
        energy = self.energy(zi)
        reg_val = self.reg_fun(zi)
        proj_val = self.proj_error(zi)
        
        return lax.stop_gradient(energy + self.lam1_norm*reg_val + self.lam2_norm*proj_val)
    
    def Dregenergy(self,
                   zi:Array,
                   ui:Array,
                   Gi:Array,
                   gi:Array,
                   )->Array:
        
        return gi+2.*(jnp.einsum('tij,tj->ti', Gi[:-1], ui[:-1])-jnp.einsum('tij,tj->ti', Gi[1:], ui[1:]))

    def inner_product(self,
                      zi:Array,
                      ui:Array,
                      )->Array:
        
        Gi = self.metric_matrix(zi)
        reg_val = self.reg_fun(zi)
        proj_val = self.proj_error(zi)
        
        return jnp.sum(jnp.einsum('...i,...ij,...j->...', ui, Gi, ui))+self.lam1_norm*reg_val \
            +self.lam2_norm*proj_val, Gi

    def gi(self,
           zi:Array,
           ui:Array,
           )->Tuple[Array]:
        
        gi, Gi = lax.stop_gradient(grad(self.inner_product, has_aux=True)(zi,ui[1:]))
        Gi = jnp.vstack((self.G0.reshape(-1,self.dim,self.dim),
                         Gi))
        
        return gi, Gi
    
    def update_scheme(self, 
                      gi:Array, 
                      Gi_inv:Array,
                      )->Array:
        
        g_cumsum = jnp.cumsum(gi[::-1], axis=0)[::-1]
        ginv_sum = jnp.sum(Gi_inv, axis=0)
        
        rhs = jnp.sum(jnp.einsum('tij,tj->ti', Gi_inv[:-1], g_cumsum), axis=0)+2.0*self.diff
        
        muN = -jnp.linalg.solve(ginv_sum, rhs)
        mui = jnp.vstack((muN+g_cumsum, muN))
        
        return mui
    
    def update_xi(self,
                  zi:Array,
                  alpha:Array,
                  ui_hat:Array,
                  ui:Array,
                  )->Array:
        
        return (self.z0+jnp.cumsum(alpha*ui_hat[:-1]+(1.-alpha)*ui[:-1], axis=0),)
    
    def cond_fun(self, 
                 carry:Tuple,
                 )->Array:
        
        zi, ui, Gi, gi, Gi_inv, grad_val, idx = carry
        
        grad_norm = jnp.linalg.norm(grad_val)

        return (grad_norm>self.tol) & (idx < self.max_iter)
    
    def georce_step(self,
                    carry:Tuple,
                    )->Array:
        
        zi, ui, Gi, gi, Gi_inv, grad_val, idx = carry
        
        mui = self.update_scheme(gi, Gi_inv)
        
        ui_hat = -0.5*jnp.einsum('tij,tj->ti', Gi_inv, mui)
        tau = self.line_search((zi,), grad_val, ui_hat, ui)

        ui = tau*ui_hat+(1.-tau)*ui
        zi = self.z0+jnp.cumsum(ui[:-1], axis=0)
        
        gi, Gi = self.gi(zi,ui)#jnp.einsum('tj,tjid,ti->td', un[1:], self.M.DG(xn[1:-1]), un[1:])
        Gi_inv = jnp.vstack((self.Ginv0, jnp.linalg.inv(Gi[1:])))
        grad_val = self.Dregenergy(zi, ui, Gi, gi)
        
        return (zi, ui, Gi, gi, Gi_inv, grad_val, idx+1)

    def __call__(self, 
                 z0:Array,
                 zN:Array,
                 )->Array:
        
        shape = z0.shape
        
        self.line_search = Backtracking(obj_fun=self.reg_energy,
                                        update_fun=self.update_xi,
                                        **self.line_search_params,
                                        )
        
        self.z0 = lax.stop_gradient(z0)
        self.zN = lax.stop_gradient(zN)
        self.diff = self.zN-self.z0
        self.dim = len(z0)
        
        self.G0 = lax.stop_gradient(self.metric_matrix(z0))
        self.Ginv0 = jnp.linalg.inv(self.G0).reshape(1,self.dim,self.dim)
        
        zi, ui = self.initialize()

        energy_init = lax.stop_gradient(self.energy(zi))
        reg_val_init = jnp.abs(lax.stop_gradient(self.reg_fun(zi)))
        proj_val_init = jnp.abs(lax.stop_gradient(self.proj_error(zi)))
        
        if reg_val_init>1e-6:
            self.lam1_norm = self.lam1*energy_init/reg_val_init
        else:
            self.lam1_norm = self.lam1
            
        if proj_val_init>1e-6:
            self.lam2_norm = self.lam2*energy_init/proj_val_init
        else:
            self.lam2_norm = self.lam2
        
        gi, Gi = self.gi(zi,ui)#jnp.einsum('tj,tjid,ti->td', un[1:], self.M.DG(xn[1:-1]), un[1:])
        Gi_inv = jnp.vstack((self.Ginv0, jnp.linalg.inv(Gi[1:])))
        grad_val = self.Dregenergy(zi, ui, Gi, gi)
        
        zi, ui, Gi, gi, gi_inv, grad_val, idx = lax.while_loop(self.cond_fun,
                                                               self.georce_step,
                                                               init_val=(zi,
                                                                         ui,
                                                                         Gi,
                                                                         gi,
                                                                         Gi_inv,
                                                                         grad_val,
                                                                         0),
                                                               )
        zi = jnp.vstack((z0, zi, zN))
        
        zi = self.proj_fun(zi)
        
        return zi.reshape(-1,*shape)

#%% Probabilistic GEORCE for Euclidean Background Metric

class ProbGEORCE_Euclidean(ABC):
    def __init__(self,
                 reg_fun:Callable,
                 init_fun:Callable[[Array, Array, int], Array]=None,
                 lam:float=1.0,
                 N:int=100,
                 tol:float=1e-4,
                 max_iter:int=1000,
                 line_search_params:Dict = {'rho': 0.5},
                 )->None:

        self.reg_fun = reg_fun
        
        self.lam = lam
        self.N = N
        self.tol = tol
        self.max_iter = max_iter
        self.line_search_params = line_search_params
        
        if init_fun is None:
            self.init_fun = lambda z0, zN, N: (zN-z0)*jnp.linspace(0.0,
                                                                   1.0,
                                                                   N,
                                                                   endpoint=False,
                                                                   dtype=z0.dtype)[1:].reshape(-1,1)+z0
        else:
            self.init_fun = init_fun
        
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

        zi = jnp.vstack((self.z0, lax.stop_gradient(zi), self.zN))
        ui = zi[1:]-zi[:-1]
        
        return jnp.sum(ui*ui)
    
    def reg_energy(self, 
                   zi:Array,
                   *args,
                   )->Array:

        reg_val = self.reg_fun(zi)
        energy = self.energy(zi)
        
        return lax.stop_gradient(energy+self.lam_norm*reg_val)

    def Dregenergy(self,
                   ui:Array,
                   gi:Array,
                   )->Array:
        
        return gi+2.*(ui[:-1]-ui[1:])

    def inner_product(self,
                      zi:Array,
                      )->Array:
        
        return self.lam_norm*self.reg_fun(zi)
    
    def gi(self,
           zi:Array,
           )->Array:
        
        return lax.stop_gradient(grad(self.inner_product)(zi))
    
    def update_xi(self,
                  zi:Array,
                  alpha:Array,
                  ui_hat:Array,
                  ui:Array,
                  )->Array:
        
        return (self.z0+jnp.cumsum(alpha*ui_hat[:-1]+(1.-alpha)*ui[:-1], axis=0),)
    
    def update_ui(self,
                  gi:Array,
                  )->Array:
        
        g_cumsum = jnp.vstack((jnp.cumsum(gi[::-1], axis=0)[::-1], jnp.zeros(self.dim)))
        g_sum = jnp.sum(g_cumsum, axis=0)/self.N
        
        return self.diff/self.N+0.5*(g_sum-g_cumsum)

    def cond_fun(self, 
                 carry:Tuple,
                 )->Array:
        
        zi, ui, gi, grad_val, idx = carry
        
        grad_norm = jnp.linalg.norm(grad_val)

        return (grad_norm>self.tol) & (idx < self.max_iter)
    
    def georce_step(self,
                    carry:Tuple,
                    )->Array:
        
        zi, ui, gi, grad_val, idx = carry

        ui_hat = self.update_ui(gi)
        tau = self.line_search((zi,), grad_val, ui_hat, ui)

        ui = tau*ui_hat+(1.-tau)*ui
        zi = self.z0+jnp.cumsum(ui[:-1], axis=0)
        
        gi = self.gi(zi)#jnp.einsum('tj,tjid,ti->td', un[1:], self.M.DG(xn[1:-1]), un[1:])
        grad_val = self.Dregenergy(ui, gi)
        
        return (zi, ui, gi, grad_val, idx+1)
    
    def __call__(self, 
                 z0:Array,
                 zN:Array,
                 )->Array:
        
        shape = z0.shape
        
        self.line_search = Backtracking(obj_fun=self.reg_energy,
                                        update_fun=self.update_xi,
                                        **self.line_search_params,
                                        )
        
        self.z0 = lax.stop_gradient(z0.reshape(-1))
        self.zN = lax.stop_gradient(zN.reshape(-1))
        self.diff = self.zN-self.z0
        self.dim = len(z0)
        
        zi, ui = self.initialize()

        energy_init = lax.stop_gradient(self.energy(zi))
        reg_val_init = jnp.abs(lax.stop_gradient(self.reg_fun(zi)))
        
        if reg_val_init>1e-6:
            self.lam_norm = self.lam*energy_init/reg_val_init
        else:
            self.lam_norm = self.lam
        
        gi = self.gi(zi)#jnp.einsum('tj,tjid,ti->td', un[1:], self.M.DG(xn[1:-1]), un[1:])
        grad_val = self.Dregenergy(ui, gi)
        
        zi, ui, gi, grad_val, idx = lax.while_loop(self.cond_fun, 
                                                   self.georce_step, 
                                                   init_val=(zi, 
                                                             ui, 
                                                             gi, 
                                                             grad_val, 
                                                             0),
                                                   )
        zi = jnp.vstack((z0, zi, zN))
            
        return zi.reshape(-1,*shape)