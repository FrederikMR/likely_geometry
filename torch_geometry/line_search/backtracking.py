#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 12:01:26 2024

@author: fmry
"""

#%% Sources

#%% Modules

import torch

from torch import Tensor

from typing import Callable
from abc import ABC
    
#%% Backtracking Line Search

class Backtracking(ABC):
    """Backtracking (soft line search) for any function and update equation

    Estimates the step size for any given optimization module using backtracking

    Attributes:
        obj_fun: The objecttive function
        update_fun: update of the parameters given state equation
        alpha: initial stepe size
        rho: backtracking parameter
        c1: parameter in Armijo condition
        max_iter: maximum number of backtracking iterations
    """
    def __init__(self,
                 obj_fun:Callable[[Tensor,...], float],
                 update_fun:Callable[[Tensor,float,...], Tensor],
                 alpha:float=1.0,
                 rho:float=0.9,
                 c1:float=0.90,
                 max_iter:int=100,
                 )->None:
        """Initializes the instance of Backtracking.

        Args:
          obj_fun: The objecttive function
          update_fun: update of the parameters given state equation
          alpha: initial stepe size
          rho: backtracking parameter
          c1: parameter in Armijo condition
          max_iter: maximum number of backtracking iterations
        """
        
        self.obj_fun = obj_fun
        self.update_fun = update_fun
        
        self.alpha = alpha
        self.rho = rho
        self.c1 = c1
        self.max_iter = max_iter
        
        self.x = None
        self.obj0 = None
        
        return
    
    @torch.no_grad()
    def armijo_condition(self, 
                         x_new:Tensor, 
                         obj:Tensor, 
                         alpha:Tensor, 
                         *args,
                         )->bool:
        """Armijo condition for improvement of objective fucntion

        Args:
          x_new: the candidate for new parameters
          obj: uthe objective value for the new parameters
          alpha: steps size
          *args: optional parameters for the objective function and update function
        Output:
          boolean indicating if improvement of objective function
        """
        
        val1 = self.obj0+self.c1*alpha*torch.sum(self.pk*self.grad0)#torch.dot(self.pk, self.grad0)
        
        return obj>val1
    
    @torch.no_grad()
    def cond_fun(self, 
                 alpha,
                 idx,
                 *args,
                 )->Tensor:
        """Condition for terminating backtracking

        Args:
          alpa: the candidate of the steps size
          idx: current iteration index
          *args: optional parameters for the objective function and update function
        Output:
          boolean indicating if backtracking should be terminated
        """

        x_new = self.update_fun(*self.x, alpha, *args)
        obj = self.obj_fun(*x_new, *args).item()
        bool_val = self.armijo_condition(x_new, obj, alpha, *args)
        
        return (bool_val) & (idx < self.max_iter)
    
    def update_alpha(self,
                     alpha:float,
                     idx:int,
                     )->Tensor:
        """Updates the step size by the backtracking parameter

        Args:
          alpa: the candidate of the steps size
          idx: current iteration index
        Output:
          alpha: updated step size 
          idx: current iteration number
        """

        return self.rho*alpha, idx+1
    
    def __call__(self, 
                 x:Tensor,
                 grad_val:Tensor,
                 *args,
                 )->Tensor:
        """Computes the updated step size alpha

        Args:
          x: current parameter
          grad_val: gradient of the objecitve function in current parameter
          *args: optional parameters for the objective function and update function
        Output:
          updated step size
        """
        
        self.x = x
        self.obj0 = self.obj_fun(*x,*args).item()
        self.pk = -grad_val
        self.grad0 = grad_val
        
        alpha, idx = self.alpha, 0
        while self.cond_fun(alpha, idx, *args):
            alpha, idx = self.update_alpha(alpha, idx)

        return alpha