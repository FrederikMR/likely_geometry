#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 12:01:26 2024

@author: fmry
"""

#%% Sources

#%% Modules

from jax_geometry.setup import *
    
#%% Backtracking Line Search

class Backtracking(ABC):
    def __init__(self,
                 obj_fun:Callable[[Array,...], Array],
                 update_fun:Callable[[Array, Array,...], Array],
                 alpha:float=1.0,
                 rho:float=0.9,
                 c1:float=0.90,
                 max_iter:int=100,
                 )->None:
        #https://optimization.cbe.cornell.edu/index.php?title=Line_search_methods
        
        self.obj_fun = obj_fun
        self.update_fun = update_fun
        
        self.alpha = alpha
        self.rho = rho
        self.c1 = c1
        self.max_iter = max_iter
        
        self.x = None
        self.obj0 = None
        
        return

    def armijo_condition(self, x_new:Array, obj:Array, alpha:Array, *args)->bool:
        
        val1 = self.obj0+self.c1*alpha*jnp.sum(self.pk*self.grad0)

        return obj>self.obj0
    
    def cond_fun(self, 
                 carry:Tuple,
                 )->Array:
        
        alpha, idx, *args = carry
        
        x_new = lax.stop_gradient(self.update_fun(*self.x, alpha, *args))
        obj = lax.stop_gradient(self.obj_fun(*x_new, *args))
        
        bool_val = self.armijo_condition(x_new, obj, alpha, *args)
        
        return (bool_val) & (idx < self.max_iter)
    
    def update_alpha(self,
                     carry:Tuple[Array, int]
                     )->Array:
        
        alpha, idx, *_ = carry
        
        return (self.rho*alpha, idx+1, *_)
    
    def __call__(self, 
                 x:Array,
                 grad_val:Array,
                 *args,
                 )->Array:
        
        self.x = x
        self.obj0 = lax.stop_gradient(self.obj_fun(*x,*args))
        self.pk = -grad_val
        self.grad0 = grad_val
        
        alpha, *_ = lax.while_loop(self.cond_fun,
                                   self.update_alpha,
                                   init_val = (self.alpha, 0, *args)
                                   )
        
        return alpha