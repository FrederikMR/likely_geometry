#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 17:06:07 2025

@author: fmry
"""

#%% Sources

#%% Modules

from jax_geometry.setup import *

from jax_geometry.manifolds import RiemannianManifold

#%% Gradient Descent Estimation of Geodesics

class ProbGEORCEFM_Adaptive(ABC):
    def __init__(self,
                 M:RiemannianManifold,
                 reg_fun:Callable[[Array],Array],
                 init_fun:Callable[[Array, Array, int], Array]=None,
                 lam:float=1.0,
                 N_grid:int=100,
                 tol:float=1e-4,
                 max_iter:int=1000,
                 lr_rate:float=0.1,
                 beta1:float=0.5,
                 beta2:float=0.5,
                 eps:float=1e-8,
                 parallel:bool=True,
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
        
        if parallel:
            self.reg_energy = self.vmap_reg_energy
            self.energy = self.vmap_energy
            self.gi = self.vmap_gi
        else:
            self.reg_energy = self.loop_reg_energy
            self.energy = self.loop_energy
            self.gi = self.loop_gi
        
        if init_fun is None:
            self.init_fun = lambda z0, zT, N_grid: (zT-z0)*jnp.linspace(0.0,
                                                                   1.0,
                                                                   N_grid,
                                                                   endpoint=False,
                                                                   dtype=z0.dtype)[1:].reshape(-1,1)+z0
        else:
            self.init_fun = init_fun
        
    def __str__(self)->str:
        
        return "Geodesic Computation Object using Control Problem"
    
    def init_curve(self, 
                   z0:Array, 
                   zT:Array,
                   )->Array:
        
        zi = self.init_fun(z0, zT, self.N_grid)
        total = jnp.vstack((z0, zi, zT))
        ui = total[1:]-total[:-1]
        
        return zi, ui
    
    def vmap_reg_energy(self, 
                        zi:Array,
                        z_mu:Array,
                        *args,
                        )->Array:

        energy = vmap(self.path_energy, in_axes=(0,0,0,None))(self.z_obs, zi, self.G0, z_mu)
        reg_val = self.wi*(vmap(self.reg_fun)(zi) + self.N_data*self.reg_fun(z_mu))

        return jnp.sum(self.wi*energy) + self.lam_norm*jnp.sum(reg_val)
    
    def loop_reg_energy(self, 
                        zi:Array,
                        z_mu:Array,
                        *args,
                        )->Array:
        
        def step_energy(energy:Array,
                        y:Tuple,
                        )->Tuple:
            
            z, z_obs, w, G0 = y
            
            energy += w*self.path_energy(z_obs, z, G0, z_mu) + w*self.lam_norm*self.reg_fun(z)

            return (energy,)*2
        
        zi = zi.reshape(self.N_data, -1, self.dim)
        
        energy, _ = lax.scan(step_energy,
                             init=0.0,
                             xs=(zi, self.z_obs, self.wi, self.G0),
                             )

        return energy + self.lam_norm*self.N_data*self.reg_fun(z_mu)
    
    def vmap_energy(self, 
                    zi:Array,
                    z_mu:Array,
                    *args,
                    )->Array:

        energy = vmap(self.path_energy, in_axes=(0,0,0,None))(self.z_obs, zi, self.G0, z_mu)

        return jnp.sum(self.wi*energy)
    
    def loop_energy(self, 
                    zi:Array,
                    z_mu:Array,
                    *args,
                    )->Array:
        
        def step_energy(energy:Array,
                        y:Tuple,
                        )->Tuple:
            
            z, z_obs, w, G0 = y
            
            energy += w*self.path_energy(z_obs, z, G0, z_mu)

            return (energy,)*2
        
        zi = zi.reshape(self.N_data, -1, self.dim)
        
        energy, _ = lax.scan(step_energy,
                             init=0.0,
                             xs=(zi, self.z_obs, self.wi, self.G0),
                             )

        return energy
    
    def path_energy(self, 
                    z0:Array,
                    zi:Array,
                    G0:Array,
                    z_mu:Array,
                    )->Array:
        
        term1 = zi[0]-z0
        val1 = jnp.einsum('i,ij,j->', term1, G0, term1)
        
        term2 = zi[1:]-zi[:-1]
        Gi = vmap(lambda z: self.M.G(z))(zi)
        val2 = jnp.einsum('ti,tij,tj->t', term2, Gi[:-1], term2)
        
        term3 = z_mu-zi[-1]
        val3 = jnp.einsum('i,ij,j->', term3, Gi[-1], term3)
        
        return val1+jnp.sum(val2)+val3
    
    def Dregenergy(self,
                   zi:Array,
                   *args,
                   )->Array:

        return grad(self.energy, argnums=0)(zi,*args)/self.N_data
        
    def Dregenergy_frechet(self,
                           zi:Array,
                           ui:Array,
                           z_mu:Array,
                           Gi:Array,
                           gi:Array,
                           )->Array:

        Gi = jnp.concatenate((self.G0.reshape(self.N_data, -1, self.dim, self.dim), 
                              Gi,
                              ),
                             axis=1)
        
        dcurve = jnp.mean(gi+2.*(jnp.einsum('...ij,...j->...i', Gi[:,:-1], ui[:,:-1])-\
                            jnp.einsum('...ij,...j->...i', Gi[:,1:], ui[:,1:])), axis=0)
        dmu = 2.*jnp.mean(jnp.einsum('...ij,...i->...j', Gi[:,-1], ui[:,-1]), axis=0)
        
        return jnp.hstack((dcurve.reshape(-1), dmu))
    
    def vmap_inner_product(self,
                           zi:Array,
                           ui:Array,
                           )->Array:
            
        Gi = vmap(vmap(self.M.G))(zi)
        reg_val = jnp.sum(vmap(self.reg_fun)(zi))

        return jnp.sum(jnp.einsum('...i,...ij,...j->...', ui, Gi, ui))+self.lam_norm*reg_val, Gi
    
    def loop_inner_product(self,
                           zi:Array,
                           ui:Array,
                           )->Array:
            
        Gi = vmap(self.M.G)(zi)
        reg_val = self.reg_fun(zi)

        return jnp.sum(jnp.einsum('...i,...ij,...j->...', ui, Gi, ui))+self.lam_norm*reg_val, Gi
    
    def vmap_gi(self,
                zi:Array,
                ui:Array,
                )->Array:
        
        gi, Gi = lax.stop_gradient(grad(self.vmap_inner_product, has_aux=True)(zi, ui))
        
        return gi, Gi, jnp.mean(jnp.sum(gi**2, axis=(1,2)))
    
    def loop_gi(self,
                zi:Array,
                ui:Array,
                )->Array:
        
        def step_gi(c:Tuple,
                    y:Tuple,
                    )->Tuple:
            
            z,u = y
            
            g, G = lax.stop_gradient(grad(self.loop_inner_product, has_aux=True)(z, u))
            
            return ((g,G),)*2
        
        _, (gi, Gi) = lax.scan(step_gi,
                               init=(jnp.zeros((self.N_grid-1, self.dim), dtype=zi.dtype),
                                     jnp.zeros((self.N_grid-1, self.dim,self.dim), dtype=zi.dtype)),
                               xs=(zi,ui),
                               )
        
        return gi, Gi, jnp.mean(jnp.sum(gi**2, axis=(1,2)))
        
    def curve_update(self, 
                     z_mu:Array,
                     g_cumsum:Array, 
                     Gi_inv:Array,
                     ginv_sum_inv:Array,
                     )->Array:
        
        diff = jnp.einsum('...,...i->...i', self.wi, z_mu-self.z_obs)
        
        rhs = jnp.sum(jnp.einsum('...ij,...j->...i', Gi_inv[:,:-1], g_cumsum), axis=1)+2.0*diff

        muT = -jnp.einsum('...ij,...j->...i', ginv_sum_inv, rhs).reshape(-1,1,self.dim)
        mui = jnp.concatenate((muT+g_cumsum, muT), axis=1)
        
        return mui
    
    def frechet_update(self,
                       g_cumsum:Array,
                       Gi_inv:Array,
                       ginv_sum_inv:Array,
                       )->Array:
        
        rhs = jnp.einsum('k,kji,ki->kj', self.wi, ginv_sum_inv, self.z_obs) \
            -0.5*jnp.einsum('kji, ki->kj', ginv_sum_inv,
                            jnp.sum(jnp.einsum('ktij,ktj->kti', Gi_inv[:,:-1], g_cumsum), axis=1),
                            )
            
        lhs = jnp.einsum('t,tij->tij', self.wi, ginv_sum_inv)

        mu = jnp.linalg.solve(jnp.sum(lhs, axis=0), 
                              jnp.sum(rhs, axis=0),
                              )
        
        return mu
    
    def update_step(self,
                    zi:Array,
                    ui:Array,
                    z_mu:Array,
                    gi_hat:Array,
                    Gi_inv:Array,
                    kappa:float,
                    )->Tuple[Array, Array]:
        
        g_cumsum = jnp.cumsum(gi_hat[:,::-1], axis=1)[:,::-1]
        ginv_sum_inv = jnp.linalg.inv(jnp.sum(Gi_inv, axis=1))
        
        z_mu_hat = self.frechet_update(g_cumsum, Gi_inv, ginv_sum_inv)
        mui = self.curve_update(z_mu_hat, g_cumsum, Gi_inv, ginv_sum_inv)
        ui_hat = -0.5*jnp.einsum('k,ktij,ktj->kti', 1./self.wi, Gi_inv, mui)

        zi_hat = self.z_obs.reshape(-1,1,self.dim)+jnp.cumsum(ui_hat[:,:-1], axis=1)

        return zi+kappa*(zi_hat-zi), ui+kappa*(ui_hat-ui), z_mu + kappa*(z_mu_hat-z_mu)
    
    def adaptive_update(self,
                        Gi_k1:Array,
                        Gi_k2:Array,
                        gi_k1:Array,
                        gi_k2:Array,
                        rg_k1:Array,
                        rg_k2:Array,
                        beta1:Array,
                        beta2:Array,
                        idx:int,
                        )->Tuple[Array, Array, Array, Array, Array, Array, Array,
                                 Array, Array, Array]:
    
        Gi_k2 = (1.-self.beta1)*Gi_k2+self.beta1*Gi_k1
        gi_k2 = (1.-self.beta1)*gi_k2+self.beta1*gi_k1
        rg_k2 = (1.-self.beta2)*rg_k2 +self.beta2*rg_k1
        
        beta1 = beta1*self.beta1
        beta2 = beta2*self.beta2

        Gi_hat = Gi_k2/(1.-beta1)
        gi_hat = gi_k2/(1.-beta1)
        vt = rg_k2/(1.-beta2)
        
        lr = self.lr_rate/(jnp.sqrt(1+vt)+self.eps)
        
        kappa = lax.select(lr > 1.0,
                           1.0,
                           lr,
                           )
        
        return Gi_k2, Gi_hat, gi_k2, gi_hat, rg_k2, beta1, beta2, kappa, idx

    def cond_fun(self, 
                 carry:Tuple[Array,Array,Array, Array, int],
                 )->Array:
        
        zi, ui, z_mu, Gi_k1, Gi_hat, gi_k1, gi_hat, rg_k1, \
            grad_norm, beta1, beta2, kappa, idx = carry

        return (grad_norm>self.tol) & (idx < self.max_iter)
    
    def while_step(self,
                     carry:Tuple[Array,Array,Array, Array, int],
                     )->Array:
        
        zi, ui, z_mu, Gi_k1, Gi_hat, gi_k1, gi_hat, rg_k1, \
            grad_norm, beta1, beta2, kappa, idx = carry
            
            
        Gi_inv = jnp.concatenate((self.Ginv0.reshape(self.N_data, -1, self.dim, self.dim), 
                                  jnp.linalg.inv(Gi_hat),
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
        
        grad_norm = jnp.linalg.norm(self.Dregenergy_frechet(zi, ui, z_mu, Gi_hat, gi_hat).reshape(-1))
        
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
                 z_obs:Array,
                 wi:Array=None,
                 z_mu_init:Array=None,
                 )->Array:
        
        self.z_obs = z_obs
        self.N_data, self.dim = self.z_obs.shape
        
        self.G0 = lax.stop_gradient(vmap(self.M.G)(self.z_obs))
        self.Ginv0 = jnp.linalg.inv(self.G0)
        
        if wi is None:
            self.wi = jnp.ones(self.N_data)
        else:
            self.wi = wi
        
        if z_mu_init is None:
            z_mu_init = jnp.mean(self.z_obs, axis=0)

        zi, ui = vmap(self.init_curve, in_axes=(0,None))(self.z_obs, z_mu_init)
        
        energy_init = lax.stop_gradient(self.energy(zi, z_mu_init))
        reg_val_init = jnp.abs(lax.stop_gradient(jnp.sum(vmap(self.reg_fun)(zi) + self.reg_fun(z_mu_init))))
        
        self.lam_norm = lax.cond(reg_val_init < 1e-6,
                                 lambda *_: self.lam,
                                 lambda *_: self.lam*energy_init/reg_val_init,
                                 )

        gi, Gi, rg = self.gi(zi, ui[:,1:])
        grad_norm = jnp.linalg.norm(self.Dregenergy_frechet(zi, ui, z_mu_init, Gi, gi).reshape(-1))
        
        zi, ui, z_mu, Gi_k1, Gi_hat, gi_k1, gi_hat, rg_k1, \
            grad_norm, beta1, beta2, kappa, idx = \
                lax.while_loop(self.cond_fun,
                               self.while_step,
                               init_val=(zi,
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
                                         ),
                               )
        
        zi = zi[:,::-1]
            
        return z_mu, zi
    
#%% Gradient Descent Estimation of Geodesics

class ProbGEORCEFM_Embedded_Adaptive(ABC):
    def __init__(self,
                 proj_fun:Callable,
                 metric_matrix:Callable,
                 reg_fun:Callable[[Array],Array],
                 init_fun:Callable[[Array, Array, int], Array]=None,
                 lam1:float=1.0,
                 lam2:float=1.0,
                 N_grid:int=100,
                 tol:float=1e-4,
                 max_iter:int=1000,
                 lr_rate:float=0.1,
                 beta1:float=0.5,
                 beta2:float=0.5,
                 eps:float=1e-8,
                 parallel:bool=True,
                 )->None:
        
        self.proj_fun = proj_fun
        self.proj_error = lambda x: jnp.sum((proj_fun(x)-x)**2)
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
        
        if parallel:
            self.reg_energy = self.vmap_reg_energy
            self.energy = self.vmap_energy
            self.gi = self.vmap_gi
        else:
            self.reg_energy = self.loop_reg_energy
            self.energy = self.loop_energy
            self.gi = self.loop_gi
        
        if init_fun is None:
            self.init_fun = lambda z0, zT, N_grid: (zT-z0)*jnp.linspace(0.0,
                                                                   1.0,
                                                                   N_grid,
                                                                   endpoint=False,
                                                                   dtype=z0.dtype)[1:].reshape(-1,1)+z0
        else:
            self.init_fun = init_fun
        
    def __str__(self)->str:
        
        return "Geodesic Computation Object using Control Problem"
    
    def init_curve(self, 
                   z0:Array, 
                   zT:Array,
                   )->Array:
        
        zi = self.init_fun(z0, zT, self.N_grid)
        total = jnp.vstack((z0, zi, zT))
        ui = total[1:]-total[:-1]
        
        return zi, ui
    
    def vmap_reg_energy(self, 
                        zi:Array,
                        z_mu:Array,
                        *args,
                        )->Array:

        energy = vmap(self.path_energy, in_axes=(0,0,0,None))(self.z_obs, zi, self.G0, z_mu)
        reg_val = self.wi*(vmap(self.reg_fun)(zi) + self.N_data*self.reg_fun(z_mu))
        proj_val = self.wi*(vmap(self.proj_error)(zi) + self.N_data*self.proj_error(z_mu))

        return jnp.sum(self.wi*energy) + self.lam1_norm*jnp.sum(reg_val) + self.lam2_norm*jnp.sum(proj_val)
    
    def loop_reg_energy(self, 
                        zi:Array,
                        z_mu:Array,
                        *args,
                        )->Array:
        
        def step_energy(energy:Array,
                        y:Tuple,
                        )->Tuple:
            
            z, z_obs, w, G0 = y
            
            energy += w*self.path_energy(z_obs, z, G0, z_mu) + w*self.lam1_norm*self.reg_fun(z) \
                +w*self.lam2_norm*self.proj_error(z)

            return (energy,)*2
        
        zi = zi.reshape(self.N_data, -1, self.dim)
        
        energy, _ = lax.scan(step_energy,
                             init=0.0,
                             xs=(zi, self.z_obs, self.wi, self.G0),
                             )

        return energy + self.lam1_norm*self.N_data*self.reg_fun(z_mu) + self.lam2_norm*self.N_data*self.proj_val(z_mu)
    
    def vmap_energy(self, 
                    zi:Array,
                    z_mu:Array,
                    *args,
                    )->Array:

        energy = vmap(self.path_energy, in_axes=(0,0,0,None))(self.z_obs, zi, self.G0, z_mu)

        return jnp.sum(self.wi*energy)
    
    def loop_energy(self, 
                    zi:Array,
                    z_mu:Array,
                    *args,
                    )->Array:
        
        def step_energy(energy:Array,
                        y:Tuple,
                        )->Tuple:
            
            z, z_obs, w, G0 = y
            
            energy += w*self.path_energy(z_obs, z, G0, z_mu)

            return (energy,)*2
        
        zi = zi.reshape(self.N_data, -1, self.dim)
        
        energy, _ = lax.scan(step_energy,
                             init=0.0,
                             xs=(zi, self.z_obs, self.wi, self.G0),
                             )

        return energy
    
    def path_energy(self, 
                    z0:Array,
                    zi:Array,
                    G0:Array,
                    z_mu:Array,
                    )->Array:
        
        term1 = zi[0]-z0
        val1 = jnp.einsum('i,ij,j->', term1, G0, term1)
        
        term2 = zi[1:]-zi[:-1]
        Gi = vmap(lambda z: self.metric_matrix(z))(zi)
        val2 = jnp.einsum('ti,tij,tj->t', term2, Gi[:-1], term2)
        
        term3 = z_mu-zi[-1]
        val3 = jnp.einsum('i,ij,j->', term3, Gi[-1], term3)
        
        return val1+jnp.sum(val2)+val3
    
    def Dregenergy(self,
                   zi:Array,
                   *args,
                   )->Array:

        return grad(self.energy, argnums=0)(zi,*args)/self.N_data
        
    def Dregenergy_frechet(self,
                           zi:Array,
                           ui:Array,
                           z_mu:Array,
                           Gi:Array,
                           gi:Array,
                           )->Array:

        Gi = jnp.concatenate((self.G0.reshape(self.N_data, -1, self.dim, self.dim), 
                              Gi,
                              ),
                             axis=1)
        
        dcurve = jnp.mean(gi+2.*(jnp.einsum('...ij,...j->...i', Gi[:,:-1], ui[:,:-1])-\
                            jnp.einsum('...ij,...j->...i', Gi[:,1:], ui[:,1:])), axis=0)
        dmu = 2.*jnp.mean(jnp.einsum('...ij,...i->...j', Gi[:,-1], ui[:,-1]), axis=0)
        
        return jnp.hstack((dcurve.reshape(-1), dmu))
    
    def vmap_inner_product(self,
                           zi:Array,
                           ui:Array,
                           )->Array:
            
        Gi = vmap(vmap(self.metric_matrix))(zi)
        reg_val = jnp.sum(vmap(self.reg_fun)(zi))
        proj_val = jnp.sum(vmap(self.proj_error)(zi))

        return jnp.sum(jnp.einsum('...i,...ij,...j->...', ui, Gi, ui))+self.lam1_norm*reg_val+self.lam2_norm*proj_val, Gi
    
    def loop_inner_product(self,
                           zi:Array,
                           ui:Array,
                           )->Array:
            
        Gi = vmap(self.metric_matrix)(zi)
        reg_val = self.reg_fun(zi)
        proj_val = self.proj_error(zi)

        return jnp.sum(jnp.einsum('...i,...ij,...j->...', ui, Gi, ui))+self.lam1_norm*reg_val + self.lam2_norm*proj_val, Gi
    
    def vmap_gi(self,
                zi:Array,
                ui:Array,
                )->Array:
        
        gi, Gi = lax.stop_gradient(grad(self.vmap_inner_product, has_aux=True)(zi, ui))
        
        return gi, Gi, jnp.mean(jnp.sum(gi**2, axis=(1,2)))
    
    def loop_gi(self,
                zi:Array,
                ui:Array,
                )->Array:
        
        def step_gi(c:Tuple,
                    y:Tuple,
                    )->Tuple:
            
            z,u = y
            
            g, G = lax.stop_gradient(grad(self.loop_inner_product, has_aux=True)(z, u))
            
            return ((g,G),)*2
        
        _, (gi, Gi) = lax.scan(step_gi,
                               init=(jnp.zeros((self.N_grid-1, self.dim), dtype=zi.dtype),
                                     jnp.zeros((self.N_grid-1, self.dim,self.dim), dtype=zi.dtype)),
                               xs=(zi,ui),
                               )
        
        return gi, Gi, jnp.mean(jnp.sum(gi**2, axis=(1,2)))
        
    def curve_update(self, 
                     z_mu:Array,
                     g_cumsum:Array, 
                     Gi_inv:Array,
                     ginv_sum_inv:Array,
                     )->Array:
        
        diff = jnp.einsum('...,...i->...i', self.wi, z_mu-self.z_obs)
        
        rhs = jnp.sum(jnp.einsum('...ij,...j->...i', Gi_inv[:,:-1], g_cumsum), axis=1)+2.0*diff

        muT = -jnp.einsum('...ij,...j->...i', ginv_sum_inv, rhs).reshape(-1,1,self.dim)
        mui = jnp.concatenate((muT+g_cumsum, muT), axis=1)
        
        return mui
    
    def frechet_update(self,
                       g_cumsum:Array,
                       Gi_inv:Array,
                       ginv_sum_inv:Array,
                       )->Array:
        
        rhs = jnp.einsum('k,kji,ki->kj', self.wi, ginv_sum_inv, self.z_obs) \
            -0.5*jnp.einsum('kji, ki->kj', ginv_sum_inv,
                            jnp.sum(jnp.einsum('ktij,ktj->kti', Gi_inv[:,:-1], g_cumsum), axis=1),
                            )
            
        lhs = jnp.einsum('t,tij->tij', self.wi, ginv_sum_inv)

        mu = jnp.linalg.solve(jnp.sum(lhs, axis=0), 
                              jnp.sum(rhs, axis=0),
                              )
        
        return mu
    
    def update_step(self,
                    zi:Array,
                    ui:Array,
                    z_mu:Array,
                    gi_hat:Array,
                    Gi_inv:Array,
                    kappa:float,
                    )->Tuple[Array, Array]:
        
        g_cumsum = jnp.cumsum(gi_hat[:,::-1], axis=1)[:,::-1]
        ginv_sum_inv = jnp.linalg.inv(jnp.sum(Gi_inv, axis=1))
        
        z_mu_hat = self.frechet_update(g_cumsum, Gi_inv, ginv_sum_inv)
        mui = self.curve_update(z_mu_hat, g_cumsum, Gi_inv, ginv_sum_inv)
        ui_hat = -0.5*jnp.einsum('k,ktij,ktj->kti', 1./self.wi, Gi_inv, mui)

        zi_hat = self.z_obs.reshape(-1,1,self.dim)+jnp.cumsum(ui_hat[:,:-1], axis=1)

        return zi+kappa*(zi_hat-zi), ui+kappa*(ui_hat-ui), z_mu + kappa*(z_mu_hat-z_mu)
    
    def adaptive_update(self,
                        Gi_k1:Array,
                        Gi_k2:Array,
                        gi_k1:Array,
                        gi_k2:Array,
                        rg_k1:Array,
                        rg_k2:Array,
                        beta1:Array,
                        beta2:Array,
                        idx:int,
                        )->Tuple[Array, Array, Array, Array, Array, Array, Array,
                                 Array, Array, Array]:
    
        Gi_k2 = (1.-self.beta1)*Gi_k2+self.beta1*Gi_k1
        gi_k2 = (1.-self.beta1)*gi_k2+self.beta1*gi_k1
        rg_k2 = (1.-self.beta2)*rg_k2 +self.beta2*rg_k1
        
        beta1 = beta1*self.beta1
        beta2 = beta2*self.beta2

        Gi_hat = Gi_k2/(1.-beta1)
        gi_hat = gi_k2/(1.-beta1)
        vt = rg_k2/(1.-beta2)
        
        lr = self.lr_rate/(jnp.sqrt(1+vt)+self.eps)
        
        kappa = lax.select(lr > 1.0,
                           1.0,
                           lr,
                           )
        
        return Gi_k2, Gi_hat, gi_k2, gi_hat, rg_k2, beta1, beta2, kappa, idx

    def cond_fun(self, 
                 carry:Tuple[Array,Array,Array, Array, int],
                 )->Array:
        
        zi, ui, z_mu, Gi_k1, Gi_hat, gi_k1, gi_hat, rg_k1, \
            grad_norm, beta1, beta2, kappa, idx = carry

        return (grad_norm>self.tol) & (idx < self.max_iter)
    
    def while_step(self,
                     carry:Tuple[Array,Array,Array, Array, int],
                     )->Array:
        
        zi, ui, z_mu, Gi_k1, Gi_hat, gi_k1, gi_hat, rg_k1, \
            grad_norm, beta1, beta2, kappa, idx = carry
            
            
        Gi_inv = jnp.concatenate((self.Ginv0.reshape(self.N_data, -1, self.dim, self.dim), 
                                  jnp.linalg.inv(Gi_hat),
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
        
        grad_norm = jnp.linalg.norm(self.Dregenergy_frechet(zi, ui, z_mu, Gi_hat, gi_hat).reshape(-1))
        
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
                 z_obs:Array,
                 wi:Array=None,
                 z_mu_init:Array=None,
                 )->Array:

        self.z_obs = z_obs
        self.N_data, self.dim = self.z_obs.shape
        
        self.G0 = lax.stop_gradient(vmap(self.metric_matrix)(self.z_obs))
        self.Ginv0 = jnp.linalg.inv(self.G0)
        
        if wi is None:
            self.wi = jnp.ones(self.N_data)
        else:
            self.wi = wi
        
        if z_mu_init is None:
            z_mu_init = jnp.mean(self.z_obs, axis=0)

        zi, ui = vmap(self.init_curve, in_axes=(0,None))(self.z_obs, z_mu_init)
        
        energy_init = lax.stop_gradient(self.energy(zi, z_mu_init))
        reg_val_init = jnp.abs(lax.stop_gradient(jnp.sum(vmap(self.reg_fun)(zi) + self.reg_fun(z_mu_init))))
        proj_val_init = jnp.abs(lax.stop_gradient(jnp.sum(vmap(self.proj_error)(zi) + self.proj_error(z_mu_init))))
        
        if reg_val_init>1e-6:
            self.lam1_norm = self.lam1*energy_init/reg_val_init
        else:
            self.lam1_norm = self.lam1
            
        if proj_val_init>1e-6:
            self.lam2_norm = self.lam2*energy_init/proj_val_init
        else:
            self.lam2_norm = self.lam2

        gi, Gi, rg = self.gi(zi, ui[:,1:])
        grad_norm = jnp.linalg.norm(self.Dregenergy_frechet(zi, ui, z_mu_init, Gi, gi).reshape(-1))
        
        zi, ui, z_mu, Gi_k1, Gi_hat, gi_k1, gi_hat, rg_k1, \
            grad_norm, beta1, beta2, kappa, idx = \
                lax.while_loop(self.cond_fun,
                               self.while_step,
                               init_val=(zi,
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
                                         ),
                               )
        
        zi = zi[:,::-1]
            
        return z_mu, zi
    
#%% Gradient Descent Estimation of Geodesics

class ProbGEORCEFM_Euclidean_Adaptive(ABC):
    def __init__(self,
                 reg_fun:Callable[[Array],Array],
                 init_fun:Callable[[Array, Array, int], Array]=None,
                 lam:float=1.0,
                 N_grid:int=100,
                 tol:float=1e-4,
                 max_iter:int=1000,
                 lr_rate:float=0.1,
                 beta1:float=0.5,
                 beta2:float=0.5,
                 eps:float=1e-8,
                 parallel:bool=True,
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
        
        if parallel:
            self.reg_energy = self.vmap_reg_energy
            self.energy = self.vmap_energy
            self.gi = self.vmap_gi
        else:
            self.reg_energy = self.loop_reg_energy
            self.energy = self.loop_energy
            self.gi = self.loop_gi
        
        if init_fun is None:
            self.init_fun = lambda z0, zT, N_grid: (zT-z0)*jnp.linspace(0.0,
                                                                   1.0,
                                                                   N_grid,
                                                                   endpoint=False,
                                                                   dtype=z0.dtype)[1:].reshape(-1,1)+z0
        else:
            self.init_fun = init_fun
        
    def __str__(self)->str:
        
        return "Geodesic Computation Object using Control Problem"
    
    def init_curve(self, 
                   z0:Array, 
                   zT:Array,
                   )->Array:
        
        zi = self.init_fun(z0, zT, self.N_grid)
        total = jnp.vstack((z0, zi, zT))
        ui = total[1:]-total[:-1]
        
        return zi, ui
    
    def vmap_reg_energy(self, 
                        zi:Array,
                        z_mu:Array,
                        *args,
                        )->Array:

        energy = vmap(self.path_energy, in_axes=(0,0,None))(self.z_obs, zi, z_mu)
        reg_val = self.wi*(vmap(self.reg_fun)(zi) + self.N_data*self.reg_fun(z_mu))

        return jnp.sum(self.wi*energy) + self.lam_norm*jnp.sum(reg_val)
    
    def loop_reg_energy(self, 
                        zi:Array,
                        z_mu:Array,
                        *args,
                        )->Array:
        
        def step_energy(energy:Array,
                        y:Tuple,
                        )->Tuple:
            
            z, z_obs, w, G0 = y
            
            energy += w*self.path_energy(z_obs, z, G0, z_mu) + w*self.lam_norm*self.reg_fun(z)

            return (energy,)*2
        
        zi = zi.reshape(self.N_data, -1, self.dim)
        
        energy, _ = lax.scan(step_energy,
                             init=0.0,
                             xs=(zi, self.z_obs, self.wi, self.G0),
                             )

        return energy + self.N_data*self.reg_fun(z_mu)
    
    def vmap_energy(self, 
                    zi:Array,
                    z_mu:Array,
                    *args,
                    )->Array:

        energy = vmap(self.path_energy, in_axes=(0,0,None))(self.z_obs, zi, z_mu)

        return jnp.sum(self.wi*energy)
    
    def loop_energy(self, 
                    zi:Array,
                    z_mu:Array,
                    *args,
                    )->Array:
        
        def step_energy(energy:Array,
                        y:Tuple,
                        )->Tuple:
            
            z, z_obs, w, G0 = y
            
            energy += w*self.path_energy(z_obs, z, z_mu)

            return (energy,)*2
        
        zi = zi.reshape(self.N_data, -1, self.dim)
        
        energy, _ = lax.scan(step_energy,
                             init=0.0,
                             xs=(zi, self.z_obs, self.wi),
                             )

        return energy
    
    def path_energy(self, 
                    z0:Array,
                    zi:Array,
                    z_mu:Array,
                    )->Array:
        
        term1 = zi[0]-z0
        val1 = jnp.einsum('i,i->', term1, term1)
        
        term2 = zi[1:]-zi[:-1]
        val2 = jnp.einsum('ti,ti->t', term2, term2)
        
        term3 = z_mu-zi[-1]
        val3 = jnp.einsum('i,i->', term3, term3)
        
        return val1+jnp.sum(val2)+val3
    
    def Dregenergy(self,
                   zi:Array,
                   *args,
                   )->Array:

        return grad(self.energy, argnums=0)(zi,*args)/self.N_data
        
    def Dregenergy_frechet(self,
                           zi:Array,
                           ui:Array,
                           z_mu:Array,
                           gi:Array,
                           )->Array:
        
        dcurve = jnp.mean(gi+2.*(ui[:,:-1]-ui[:,1:]), axis=0)
        dmu = 2.*jnp.mean(ui[:,-1], axis=0)
        
        return jnp.hstack((dcurve.reshape(-1), dmu))
    
    def vmap_inner_product(self,
                           zi:Array,
                           )->Array:

        reg_val = jnp.sum(vmap(self.reg_fun)(zi))

        return self.lam_norm*reg_val
    
    def loop_inner_product(self,
                           zi:Array,
                           )->Array:

        reg_val = self.reg_fun(zi)

        return self.lam_norm*reg_val
    
    def vmap_gi(self,
                zi:Array,
                )->Array:
        
        gi = lax.stop_gradient(grad(self.vmap_inner_product)(zi))
        
        return gi, jnp.mean(jnp.sum(gi**2, axis=(1,2)))
    
    def loop_gi(self,
                zi:Array,
                )->Array:
        
        def step_gi(g:Array,
                    z:Array,
                    )->Tuple:
            
            g = lax.stop_gradient(grad(self.loop_inner_product)(z))
            
            return (g,)*2
        
        _, gi = lax.scan(step_gi,
                         init=jnp.zeros((self.N_grid-1, self.dim), dtype=zi.dtype),
                         xs=(zi),
                         )
        
        return gi, jnp.mean(jnp.sum(gi**2, axis=(1,2)))
        
    def curve_update(self, 
                     z_mu:Array,
                     g_cumsum:Array, 
                     )->Array:

        g_cumsum = jnp.concatenate((g_cumsum, jnp.zeros((self.N_data, 1, self.dim))), axis=1)
        term1 = 0.5*(jnp.sum(g_cumsum, axis=1)[:,None,:]/self.N_grid-g_cumsum)/self.wi[:,None,None]
        term2 = (z_mu - self.z_obs)/self.N_grid
        
        return term1 + term2[:,None,:]
    
    def frechet_update(self,
                       g_cumsum:Array,
                       )->Array:

        sum_w = jnp.sum(self.wi)
        term1 = jnp.sum(self.wi[:,None]*self.z_obs, axis=0) - 0.5*jnp.sum(g_cumsum, axis=(0,1))
        
        return term1/sum_w
    
    def update_step(self,
                    zi:Array,
                    ui:Array,
                    z_mu:Array,
                    gi_hat:Array,
                    kappa:float,
                    )->Tuple[Array, Array]:
        
        g_cumsum = jnp.cumsum(gi_hat[:,::-1], axis=1)[:,::-1]
        
        z_mu_hat = self.frechet_update(g_cumsum)
        ui_hat = self.curve_update(z_mu_hat, g_cumsum)

        zi_hat = self.z_obs.reshape(-1,1,self.dim)+jnp.cumsum(ui_hat[:,:-1], axis=1)

        return zi+kappa*(zi_hat-zi), ui+kappa*(ui_hat-ui), z_mu + kappa*(z_mu_hat-z_mu)
    
    def adaptive_update(self,
                        gi_k1:Array,
                        gi_k2:Array,
                        rg_k1:Array,
                        rg_k2:Array,
                        beta1:Array,
                        beta2:Array,
                        idx:int,
                        )->Tuple[Array, Array, Array, Array, Array, Array, Array,
                                 Array, Array, Array]:
    
        gi_k2 = (1.-self.beta1)*gi_k2+self.beta1*gi_k1
        rg_k2 = (1.-self.beta2)*rg_k2 +self.beta2*rg_k1
        
        beta1 = beta1*self.beta1
        beta2 = beta2*self.beta2

        gi_hat = gi_k2/(1.-beta1)
        vt = rg_k2/(1.-beta2)
        
        lr = self.lr_rate/(jnp.sqrt(1+vt)+self.eps)
        
        kappa = lax.select(lr > 1.0,
                           1.0,
                           lr,
                           )
        
        return gi_k2, gi_hat, rg_k2, beta1, beta2, kappa, idx

    def cond_fun(self, 
                 carry:Tuple[Array,Array,Array, Array, int],
                 )->Array:
        
        zi, ui, z_mu, gi_k1, gi_hat, rg_k1, \
            grad_norm, beta1, beta2, kappa, idx = carry

        return (grad_norm>self.tol) & (idx < self.max_iter)
    
    def while_step(self,
                     carry:Tuple[Array,Array,Array, Array, int],
                     )->Array:
        
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
        
        grad_norm = jnp.linalg.norm(self.Dregenergy_frechet(zi, ui, z_mu, gi_hat).reshape(-1))
        
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
                 z_obs:Array,
                 wi:Array=None,
                 z_mu_init:Array=None,
                 )->Array:
        
        self.z_obs = z_obs
        self.N_data, self.dim = self.z_obs.shape
        
        if wi is None:
            self.wi = jnp.ones(self.N_data)
        else:
            self.wi = wi
        
        if z_mu_init is None:
            z_mu_init = jnp.mean(self.z_obs, axis=0)

        zi, ui = vmap(self.init_curve, in_axes=(0,None))(self.z_obs, z_mu_init)
        
        energy_init = lax.stop_gradient(self.energy(zi, z_mu_init))
        reg_val_init = jnp.abs(lax.stop_gradient(jnp.sum(vmap(self.reg_fun)(zi) + self.reg_fun(z_mu_init))))
        
        self.lam_norm = lax.cond(reg_val_init < 1e-6,
                                 lambda *_: self.lam,
                                 lambda *_: self.lam*energy_init/reg_val_init,
                                 )

        gi, rg = self.gi(zi)
        grad_norm = jnp.linalg.norm(self.Dregenergy_frechet(zi, ui, z_mu_init, gi).reshape(-1))
        
        zi, ui, z_mu, gi_k1, gi_hat, rg_k1, \
            grad_norm, beta1, beta2, kappa, idx = \
                lax.while_loop(self.cond_fun,
                               self.while_step,
                               init_val=(zi,
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
                                         ),
                               )
        
        zi = zi[:,::-1]
            
        return z_mu, zi