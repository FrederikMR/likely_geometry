#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 11:55:45 2025

@author: fmry
"""

#%% Modules

# Import required packages
import torch
import torch.nn as nn

import normflows as nf
from torchebm.core import BaseEnergyFunction
from torchebm.datasets import CheckerboardDataset

import math

import argparse

import timeit

import os
import pickle

from typing import List, Any

from torch_geometry.manifolds import LambdaManifold, nEuclidean, FisherRao, JacobianMetric, Spherical, Linear, InverseDensity, GenerativeMetric, MongeMetric
from torch_geometry.prob_geodesics import TorchOptimizers_Euclidean
from torch_geometry.prob_geodesics import ProbGEORCE_Adaptive, ProbGEORCE_Euclidean_Adaptive, ProbGEORCE_Euclidean_Adaptive_FixedLam, ProbGEORCE_Euclidean

from torch_geometry.prob_means import ProbGEORCEFM_Adaptive, ProbGEORCEFM_Euclidean_Adaptive, ProbGEORCEFM_Euclidean

#%% Arg parse

def parse_args():
    parser = argparse.ArgumentParser()
    # File-paths
    parser.add_argument('--runtime_type', default="metrics", #metrics, grid, runtime
                        type=str)
    parser.add_argument('--model_type', default="ebm", #ebm, nf, ar, vae
                        type=str)
    parser.add_argument('--computation', default="bvp", #ivp, bvp, mean
                        type=str)
    parser.add_argument('--method', default="Monge-Metric", #ProbGEORCE, Linear, SLERP, Fisher-Rao, Fisher-Rao-Reg, Jacobian-Metric, Jacobian-Metric-Reg, Inverse-Density, Generative-Metric, Monge-Metric
                        type=str)
    parser.add_argument('--geodesic_method', default="ProbGEORCE_Adaptive", #ProbGEORCE_LS, ProbGEORCE_Adaptive, Adam, SGD, RMSprop, AdamW, LBFGS
                        type=str)
    parser.add_argument('--lam', default=20.0, #20.0
                        type=float)
    parser.add_argument('--lam_identity', default=10.0,
                        type=float)
    parser.add_argument('--grid_size', default=10,
                        type=int)
    parser.add_argument('--N_grid', default=100,
                        type=int)
    parser.add_argument('--max_iter', default=1000,
                        type=int)
    parser.add_argument('--tol', default=1e-4,
                        type=float)
    parser.add_argument('--number_repeats', default=2,
                        type=int)
    parser.add_argument('--timing_repeats', default=2,
                        type=int)
    parser.add_argument('--seed', default=2712,
                        type=int)
    parser.add_argument('--device', default='cpu',
                        type=str)
    parser.add_argument('--model_path', default='../models/gen2d/',
                        type=str)
    parser.add_argument('--save_path', default='../gen2d_results/cpu/',
                        type=str)

    args = parser.parse_args()
    return args


#%% load modules

#error to a unit circle
def reg_fun(x, prob_model):

    shape = x.shape

    if x.ndim > 1:
        return -torch.sum(prob_model(x.reshape(-1,shape[-1]))).squeeze()
    else:
        return -torch.sum(prob_model(x.reshape(-1,len(x)))).squeeze()


def load_ebm(model_path:str='models/', device:str="cpu"):
    
    dataset = CheckerboardDataset(
        n_samples = 100,
        range_limit = 2.0,
        noise = 0.05,
        device=device,
        seed = 2712
        )
    
    save_path = ''.join((model_path, 'ebm/checkboard.pt'))
    INPUT_DIM = 2
    HIDDEN_DIM = 128
    checkpoint = torch.load(save_path, weights_only=False, map_location=torch.device(device))
    
    class MLPEnergy(BaseEnergyFunction):
        """A simple MLP to act as the energy function."""
    
        def __init__(self, input_dim: int, hidden_dim: int = 64):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),  # Output a single scalar energy value
            )
    
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.network(x).squeeze(-1)
    
    ebm_model = MLPEnergy(INPUT_DIM, HIDDEN_DIM).to(device)
    
    # Load state dict
    ebm_model.load_state_dict(checkpoint['model_state_dict'])
    
    ebm_reg_fun = lambda x: reg_fun(x, lambda z: -ebm_model(z))
    
    # Coordinates
    x1 = torch.tensor([-1.5, -0.5], device=device)   # top of moon 1
    x2 = torch.tensor([2.0, 1.0], device=device)  # bottom of moon 2
    data_sample = dataset.get_data().to(device)
    
    #grid
    x_coords = torch.linspace(-3.0, 3.0, args.grid_size, device=args.device)
    y_coords = torch.linspace(-3.0, 3.0, args.grid_size, device=args.device)
    ebm_xv, ebm_yv = torch.meshgrid(x_coords, y_coords, indexing="xy")
    ebm_grid = torch.stack([ebm_xv.flatten(), ebm_yv
                            .flatten()], dim=1)
    
    return ebm_reg_fun, x1, x2, data_sample, ebm_grid 

def load_nf(model_path:str='models/', device:str="cpu"):
    
    target = nf.distributions.TwoMoons()
    
    save_path = ''.join((model_path, 'nf/two_moons.pt'))
    checkpoint = torch.load(save_path, weights_only=False, map_location=torch.device('cpu'))
    
    # Reconstruct the model architecture first
    base = nf.distributions.base.DiagGaussian(2)
    flows = []
    for i in range(32):
        param_map = nf.nets.MLP([1, 64, 64, 2], init_zeros=True)
        flows.append(nf.flows.AffineCouplingBlock(param_map))
        flows.append(nf.flows.Permute(2, mode='swap'))
    nf_model = nf.NormalizingFlow(base, flows).to(args.device)
    
    # Load state dict
    nf_model.load_state_dict(checkpoint['model_state_dict'])
    
    nf_reg_fun = lambda x: reg_fun(x, lambda z: nf_model.log_prob(z))
    
    # Coordinates
    x1 = torch.tensor([1.6, 1.2], device=device)  # top of moon 2
    x2 = torch.tensor([1.6, -1.2], device=device)  # bottom of moon 2
    
    data_sample = target.sample(100).to(device)
    
    #grid
    nf_xv, nf_yv = torch.meshgrid(torch.linspace(-3., 3., args.grid_size), torch.linspace(-3., 3., args.grid_size))
    nf_grid_size = torch.cat([nf_xv.unsqueeze(2), nf_yv.unsqueeze(2)], 2).view(-1, 2)
    nf_grid_size = nf_grid_size.to(args.device)
    
    return nf_reg_fun, x1, x2, data_sample, nf_grid_size

def load_ar(model_path:str='models/', device:str="cpu"):
    
    class SimpleAutoregressive2D(nn.Module):
        def __init__(self, hidden_dim=64):
            super().__init__()
            # p(x1): learnable Gaussian parameters
            self.x1_mean = nn.Parameter(torch.zeros(1))
            self.x1_logstd = nn.Parameter(torch.zeros(1))
    
            # p(x2 | x1): MLP outputs mean and logstd
            self.net = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 2)
            )
    
        def forward(self, x):
            x1, x2 = x[:, 0:1], x[:, 1:2]
            # log p(x1)
            logp_x1 = (-0.5 * ((x1 - self.x1_mean)**2 / self.x1_logstd.exp()**2)
                       - self.x1_logstd
                       - 0.5 * torch.log(torch.tensor(2 * torch.pi))).sum(-1)
    
            # log p(x2 | x1)
            out = self.net(x1)
            mean2, logstd2 = out[:, 0:1], out[:, 1:2]
            logp_x2_given_x1 = (-0.5 * ((x2 - mean2)**2 / logstd2.exp()**2)
                                - logstd2
                                - 0.5 * torch.log(torch.tensor(2 * torch.pi))).sum(-1)
            return logp_x1 + logp_x2_given_x1
    
        def sample(self, n=1000):
            # sample x1 ~ N(mean1, std1)
            x1 = self.x1_mean + self.x1_logstd.exp() * torch.randn(n, 1)
            # sample x2 | x1
            out = self.net(x1)
            mean2, logstd2 = out[:, 0:1], out[:, 1:2]
            x2 = mean2 + logstd2.exp() * torch.randn(n, 1)
            return torch.cat([x1, x2], dim=1)
        
    save_path = ''.join((model_path, 'ar/sinus.pt'))
    checkpoint = torch.load(save_path, weights_only=False, map_location=torch.device(device))
    
    ar_model = SimpleAutoregressive2D().to(device)
    
    # Load state dict
    ar_model.load_state_dict(checkpoint['model_state_dict'])
    
    ar_reg_fun = lambda x: reg_fun(x, lambda z: ar_model(z))
    
    x1 = torch.tensor([-0.5, math.sin(-2.0)], device=device)   # top of moon 1
    x2 = torch.tensor([2.0, math.sin(2.0)], device=device)  # bottom of moon 2
    
    data_sample = torch.stack([torch.linspace(0.0, 2.0, 100, device=device), torch.sin(torch.linspace(-1.0, 2.0, 100, device=device))]).T

    #grid
    x1_lin = torch.linspace(-2.0, 2.5, args.grid_size)
    x2_lin = torch.linspace(-2.0, 2.5, args.grid_size)
    ar_xv, ar_yv = torch.meshgrid(x1_lin, x2_lin, indexing="xy")
    ar_grid = torch.stack([ar_xv.flatten(), ar_yv.flatten()], dim=-1)
    
    return ar_reg_fun, x1, x2, data_sample, ar_grid

def load_vae(model_path:str='models/', device:str="cpu"):
    
    class VAE_3d(nn.Module):
        def __init__(self,
                     fc_h: List[int] = [3, 100],
                     fc_g: List[int] = [2, 100, 3],
                     fc_mu: List[int] = [100, 2],
                     fc_var: List[int] = [100, 2],
                     fc_h_act: List[Any] = [nn.ELU],
                     fc_g_act: List[Any] = [nn.ELU, nn.Identity],
                     fc_mu_act: List[Any] = [nn.Identity],
                     fc_var_act: List[Any] = [nn.Sigmoid]
                     ):
            super(VAE_3d, self).__init__()
        
            self.fc_h = fc_h
            self.fc_g = fc_g
            self.fc_mu = fc_mu
            self.fc_var = fc_var
            self.fc_h_act = fc_h_act
            self.fc_g_act = fc_g_act
            self.fc_mu_act = fc_mu_act
            self.fc_var_act = fc_var_act
            
            self.num_fc_h = len(fc_h)
            self.num_fc_g = len(fc_g)
            self.num_fc_mu = len(fc_mu)
            self.num_fc_var = len(fc_var)
            
            self.encoder = self.encode()
            self.mu_net = self.mu_layer()
            self.var_net = self.var_layer()
            self.decoder = self.decode()
            
            # for the gaussian likelihood
            self.log_scale = nn.Parameter(torch.Tensor([0.0]))
        
        def encode(self):
            
            layer = []
            
            for i in range(1, self.num_fc_h):
                layer.append(nn.Linear(self.fc_h[i-1], self.fc_h[i]))
                layer.append(self.fc_h_act[i-1]())
                #input_layer.append(self.activations_h[i](inplace=True))
                
            return nn.Sequential(*layer)
        
        def mu_layer(self):
            
            layer = []
            
            for i in range(1, self.num_fc_mu):
                layer.append(nn.Linear(self.fc_mu[i-1], self.fc_mu[i]))
                layer.append(self.fc_mu_act[i-1]())
                
            return nn.Sequential(*layer)
        
        def var_layer(self):
            
            layer = []
            
            for i in range(1, self.num_fc_var):
                layer.append(nn.Linear(self.fc_var[i-1], self.fc_var[i]))
                layer.append(self.fc_var_act[i-1]())
                
            return nn.Sequential(*layer)
        
        def rep_par(self, mu, std):
            
            eps = torch.randn_like(std)
            z = mu + (eps * std)
            return z
            
        def decode(self):
            
            layer = []
            
            for i in range(1, self.num_fc_g):
                layer.append(nn.Linear(self.fc_g[i-1], self.fc_g[i]))
                layer.append(self.fc_g_act[i-1]())
                
            return nn.Sequential(*layer)
        
        def gaussian_likelihood(self, x_hat, logscale, x):
            scale = torch.exp(logscale)
            mean = x_hat
            dist = torch.distributions.Normal(mean, scale)
    
            # measure prob of seeing image under p(x|z)
            log_pxz = dist.log_prob(x)
            
            return log_pxz.sum(dim=1)
    
        def kl_divergence(self, z, mu, std):
            # --------------------------
            # Monte carlo KL divergence
            # --------------------------
            # 1. define the first two probabilities (in this case Normal for both)
            p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
            q = torch.distributions.Normal(mu, std)
    
            # 2. get the probabilities from the equation
            log_qzx = q.log_prob(z)
            log_pz = p.log_prob(z)
    
            # kl
            kl = (log_qzx - log_pz)
            kl = kl.sum(-1)
            
            return kl
        
        def forward(self, x):
            
            x_encoded = self.encoder(x)
            mu, var = self.mu_net(x_encoded), self.var_net(x_encoded)
            std = torch.sqrt(var)
    
            z = self.rep_par(mu, std)
            x_hat = self.decoder(z)
                    
            # compute the ELBO with and without the beta parameter: 
            # `L^\beta = E_q [ log p(x|z) - \beta * D_KL(q(z|x) | p(z))`
            # where `D_KL(q(z|x) | p(z)) = log q(z|x) - log p(z)`
            kld = self.kl_divergence(z, mu, std)
            rec_loss = self.gaussian_likelihood(x_hat, self.log_scale, x)
            
            # elbo
            elbo = (kld - rec_loss)
            elbo = elbo.mean()
            
            return z, x_hat, mu, std, kld.mean(), -rec_loss.mean(), elbo
        
        def h(self, x):
            
            x_encoded = self.encoder(x)
            mu, var = self.mu_net(x_encoded), self.var_net(x_encoded)
            std = torch.sqrt(var)
            
            z = self.rep_par(mu, std)
            
            return mu
            
        def g(self, z):
            
            x_hat = self.decoder(z)
            
            return x_hat
    
        def elbo_latent(self, z, n_samples:int=10):

            elbo = []
            for i in range(n_samples):
                x = self.decoder(z)
            
                x_encoded = self.encoder(x)
                mu, var = self.mu_net(x_encoded), self.var_net(x_encoded)
                std = torch.sqrt(var)
        
                z = self.rep_par(mu, std)
                x_hat = self.decoder(z)
                        
                # compute the ELBO with and without the beta parameter: 
                # `L^\beta = E_q [ log p(x|z) - \beta * D_KL(q(z|x) | p(z))`
                # where `D_KL(q(z|x) | p(z)) = log q(z|x) - log p(z)`
                kld = self.kl_divergence(z, mu, std)
                rec_loss = self.gaussian_likelihood(x_hat, self.log_scale, x)
                
                # elbo
                elbo.append((kld - rec_loss))

            elbo = torch.mean(torch.stack(elbo), axis=0)
            
            return elbo

        def elbo_embedded(self, x, n_samples:int=10):

            elbo = []
            for i in range(n_samples):
                x_encoded = self.encoder(x)
                mu, var = self.mu_net(x_encoded), self.var_net(x_encoded)
                std = torch.sqrt(var)
        
                z = self.rep_par(mu, std)
                x_hat = self.decoder(z)
                        
                # compute the ELBO with and without the beta parameter: 
                # `L^\beta = E_q [ log p(x|z) - \beta * D_KL(q(z|x) | p(z))`
                # where `D_KL(q(z|x) | p(z)) = log q(z|x) - log p(z)`
                kld = self.kl_divergence(z, mu, std)
                rec_loss = self.gaussian_likelihood(x_hat, self.log_scale, x)
                
                # elbo
                elbo.append((kld - rec_loss))

            elbo = torch.mean(torch.stack(elbo), axis=0)
            
            return elbo
        
    save_path = ''.join((model_path, 'vae/circle.pt'))
    checkpoint = torch.load(save_path, weights_only=False, map_location=torch.device(device))
    
    vae_model = VAE_3d().to(device)
    
    # Load state dict
    vae_model.load_state_dict(checkpoint['model_state_dict'])
    
    vae_reg_fun = lambda x: reg_fun(x, lambda z: -vae_model.elbo_latent(z))
    
    x1 = torch.tensor([math.cos(1.5*math.pi/3), math.sin(1.5*math.pi/3), 0.0], device=device)   # top of moon 1
    x2 = torch.tensor([math.cos(1.5*math.pi), math.sin(1.5*math.pi), 0.0], device=device)  # bottom of moon 2
    
    x1 = vae_model.h(x1)    
    x2 = vae_model.h(x2)
    
    grid = torch.linspace(-2*torch.pi,2*torch.pi,100, device=device)
    dataz = torch.vstack([torch.cos(grid), torch.sin(grid), torch.zeros_like(grid)]).T   # top of moon 1
    data_sample = vae_model.h(dataz)
    
    #grid
    x_coords = torch.linspace(-1.5, 1.5, args.grid_size, device=args.device)
    y_coords = torch.linspace(-1.5, 1.5, args.grid_size, device=args.device)
    vae_xv, vae_yv = torch.meshgrid(x_coords, y_coords, indexing="xy")
    vae_grid = torch.stack([vae_xv.flatten(), vae_yv.flatten(), torch.zeros_like(vae_yv.flatten())], dim=1)
    vae_grid = vae_model.h(vae_grid)

    return vae_reg_fun, x1, x2, data_sample, vae_grid

def load_model(args):
    
    if args.model_type == "ebm":
        return load_ebm(args.model_path, args.device)
    elif args.model_type == "nf":
        return load_nf(args.model_path, args.device)
    elif args.model_type == "ar":
        return load_ar(args.model_path, args.device)
    elif args.model_type == "vae":
        return load_vae(args.model_path, args.device)
    else:
        raise ValueError(f"Invalid model type: {args.model}")
        
#%% Load geodesic methods

def load_geodesic_method(args, reg_eval):

    if args.geodesic_method == "ProbGEORCE_Adaptive":
        return ProbGEORCE_Euclidean_Adaptive(reg_fun=reg_eval,
                                             init_fun=None,
                                             lam=args.lam,
                                             N=args.N_grid,
                                             tol=args.tol,
                                             max_iter=args.max_iter,
                                             lr_rate=0.01,
                                             beta1=0.5,
                                             beta2=0.5,
                                             eps=1e-8,
                                             device=args.device,
                                             )  
    elif args.geodesic_method == "ProbGEORCE_LS":
        return ProbGEORCE_Euclidean(reg_fun=reg_eval,
                                    init_fun=None,
                                    lam=args.lam,
                                    N=args.N_grid,
                                    tol=args.tol,
                                    max_iter=args.max_iter,
                                    line_search_params={'rho': 0.5},
                                    device=args.device,
                                    )  
    else:
        try:
            optimizer_class = getattr(torch.optim, args.geodesic_method)
            
            return TorchOptimizers_Euclidean(reg_fun=reg_eval,
                                             init_fun=None,
                                             lam=args.lam,
                                             N=args.N_grid,
                                             tol=args.tol,
                                             max_iter=args.max_iter,
                                             optimizer_class=optimizer_class,
                                             lr=0.01,
                                             optimizer_kwargs=None,
                                             device=args.device,
                                             )
        except:
            raise ValueError(f"Invalid geodesic method: {args.geodesic_method}")

#%% Load methods

def load_prob_georce_adaptive(args, reg_eval):
    
    M = nEuclidean(dim=2)
    Mlambda = LambdaManifold(M=M, S=reg_eval, lam=args.lam)    
    bvp_geodesic = load_geodesic_method(args, reg_eval)

    if args.geodesic_method == "ProbGEORCE_Adaptive":
        prob_mean_adaptive = ProbGEORCEFM_Euclidean_Adaptive(reg_fun=reg_eval ,
                                                             init_fun=None,
                                                             lam=args.lam,
                                                             N_grid=args.N_grid,
                                                             tol=args.tol,
                                                             max_iter=args.max_iter,
                                                             lr_rate=0.01,
                                                             beta1=0.5,
                                                             beta2=0.5,
                                                             eps=1e-8,
                                                             device=args.device,
                                                             )
    elif args.geodesic_method == "ProbGEORCE_LS":
        prob_mean_adaptive = ProbGEORCEFM_Euclidean(reg_fun=reg_eval ,
                                                    init_fun=None,
                                                    lam=args.lam,
                                                    N_grid=args.N_grid,
                                                    tol=args.tol,
                                                    max_iter=args.max_iter,
                                                    line_search_params={'rho': 0.5},
                                                    device=args.device,
                                                    )
    else:
        prob_mean_adaptive = ProbGEORCEFM_Euclidean_Adaptive(reg_fun=reg_eval ,
                                                             init_fun=None,
                                                             lam=args.lam,
                                                             N_grid=args.N_grid,
                                                             tol=args.tol,
                                                             max_iter=args.max_iter,
                                                             lr_rate=0.01,
                                                             beta1=0.5,
                                                             beta2=0.5,
                                                             eps=1e-8,
                                                             device=args.device,
                                                             )
    
    
    
    return Mlambda.Exp_ode, bvp_geodesic, prob_mean_adaptive

def load_linear_interpolation(args, reg_eval):
    
    M = Linear()
    
    return lambda x,v: M.ivp_geodesic(x,v,args.N_grid), lambda x1,x2: M.bvp_geodesic(x1,x2,args.N_grid), lambda data: M.mean_com(data, args.N_grid)

def load_spherical_interpolation(args, reg_eval):
    
    M = Spherical(eps=1e-8)
    
    return lambda x,v: M.ivp_geodesic(x,v,args.N_grid), lambda x1,x2: M.bvp_geodesic(x1,x2,args.N_grid), lambda data: M.mean_com(data, args.N_grid)

def load_fisher_rao(args, reg_eval):
    
    if "Reg" in args.method:
        M = FisherRao(log_prob = lambda x: -reg_eval(x), dim=2, lam=args.lam_identity)
    else:
        M = FisherRao(log_prob = lambda x: -reg_eval(x), dim=2, lam=1e-4)
        
    bvp_geodesic = ProbGEORCE_Adaptive(M=M,
                                       reg_fun=lambda x: torch.zeros(1, device=args.device).squeeze(),
                                       init_fun=None,
                                       lam=0.0,
                                       N=args.N_grid,
                                       tol=args.tol,
                                       max_iter=args.max_iter,
                                       lr_rate=0.01,
                                       beta1=0.5,
                                       beta2=0.5,
                                       eps=1e-8,
                                       device=args.device,
                                       )   
    
    mean_com = ProbGEORCEFM_Adaptive(M=M,
                                     reg_fun=lambda x: torch.zeros(1, device=args.device).squeeze(),
                                     init_fun=None,
                                     lam=0.0,
                                     N_grid=args.N_grid,
                                     tol=args.tol,
                                     max_iter=args.max_iter,
                                     lr_rate=0.01,
                                     beta1=0.5,
                                     beta2=0.5,
                                     eps=1e-8,
                                     device=args.device,
                                     )
    
    return M.Exp_ode, bvp_geodesic, mean_com

def load_jacobian_met(args, reg_eval):
    
    if "Reg" in args.method:
        M = JacobianMetric(log_prob = lambda x: -reg_eval(x), dim=2, lam=args.lam_identity)
    else:
        M = JacobianMetric(log_prob = lambda x: -reg_eval(x), dim=2, lam=1e-4)
    
    bvp_geodesic = ProbGEORCE_Adaptive(M=M,
                                       reg_fun=lambda x: torch.zeros(1, device=args.device).squeeze(),
                                       init_fun=None,
                                       lam=0.0,
                                       N=args.N_grid,
                                       tol=args.tol,
                                       max_iter=args.max_iter,
                                       lr_rate=0.01,
                                       beta1=0.5,
                                       beta2=0.5,
                                       eps=1e-8,
                                       device=args.device,
                                       )   
    
    mean_com = ProbGEORCEFM_Adaptive(M=M,
                                     reg_fun=lambda x: torch.zeros(1, device=args.device).squeeze(),
                                     init_fun=None,
                                     lam=0.0,
                                     N_grid=args.N_grid,
                                     tol=args.tol,
                                     max_iter=args.max_iter,
                                     lr_rate=0.01,
                                     beta1=0.5,
                                     beta2=0.5,
                                     eps=1e-8,
                                     device=args.device,
                                     )
    
    return M.Exp_ode, bvp_geodesic, mean_com

def load_generative_met(args, reg_eval):
    
    M = GenerativeMetric(log_prob = lambda x: -reg_eval(x), dim=2, lam=1.0, p0=1.0)
    
    bvp_geodesic = ProbGEORCE_Adaptive(M=M,
                                       reg_fun=lambda x: torch.zeros(1, device=args.device).squeeze(),
                                       init_fun=None,
                                       lam=0.0,
                                       N=args.N_grid,
                                       tol=args.tol,
                                       max_iter=args.max_iter,
                                       lr_rate=0.01,
                                       beta1=0.5,
                                       beta2=0.5,
                                       eps=1e-8,
                                       device=args.device,
                                       )    
    
    mean_com = ProbGEORCEFM_Adaptive(M=M,
                                     reg_fun=lambda x: torch.zeros(1, device=args.device).squeeze(),
                                     init_fun=None,
                                     lam=0.0,
                                     N_grid=args.N_grid,
                                     tol=args.tol,
                                     max_iter=args.max_iter,
                                     lr_rate=0.01,
                                     beta1=0.5,
                                     beta2=0.5,
                                     eps=1e-8,
                                     device=args.device,
                                     )
    
    return M.Exp_ode, bvp_geodesic, mean_com

def load_inverse_met(args, reg_eval):
    
    M = InverseDensity(log_prob = lambda x: -reg_eval(x), dim=2)
    
    bvp_geodesic = ProbGEORCE_Adaptive(M=M,
                                       reg_fun=lambda x: torch.zeros(1, device=args.device).squeeze(),
                                       init_fun=None,
                                       lam=0.0,
                                       N=args.N_grid,
                                       tol=args.tol,
                                       max_iter=args.max_iter,
                                       lr_rate=0.01,
                                       beta1=0.5,
                                       beta2=0.5,
                                       eps=1e-8,
                                       device=args.device,
                                       )    
    
    mean_com = ProbGEORCEFM_Adaptive(M=M,
                                     reg_fun=lambda x: torch.zeros(1, device=args.device).squeeze(),
                                     init_fun=None,
                                     lam=0.0,
                                     N_grid=args.N_grid,
                                     tol=args.tol,
                                     max_iter=args.max_iter,
                                     lr_rate=0.01,
                                     beta1=0.5,
                                     beta2=0.5,
                                     eps=1e-8,
                                     device=args.device,
                                     )
    
    return M.Exp_ode, bvp_geodesic, mean_com

def load_monge_met(args, reg_eval):
    
    M = MongeMetric(log_prob = lambda x: -reg_eval(x), dim=2, alpha=1.0)
    
    bvp_geodesic = ProbGEORCE_Adaptive(M=M,
                                       reg_fun=lambda x: torch.zeros(1, device=args.device).squeeze(),
                                       init_fun=None,
                                       lam=0.0,
                                       N=args.N_grid,
                                       tol=args.tol,
                                       max_iter=args.max_iter,
                                       lr_rate=0.01,
                                       beta1=0.5,
                                       beta2=0.5,
                                       eps=1e-8,
                                       device=args.device,
                                       )    
    
    mean_com = ProbGEORCEFM_Adaptive(M=M,
                                     reg_fun=lambda x: torch.zeros(1, device=args.device).squeeze(),
                                     init_fun=None,
                                     lam=0.0,
                                     N_grid=args.N_grid,
                                     tol=args.tol,
                                     max_iter=args.max_iter,
                                     lr_rate=0.01,
                                     beta1=0.5,
                                     beta2=0.5,
                                     eps=1e-8,
                                     device=args.device,
                                     )
    
    return M.Exp_ode, bvp_geodesic, mean_com

def load_method(args, reg_eval):
    
    if args.method == "ProbGEORCE":
        return load_prob_georce_adaptive(args, reg_eval)
    elif args.method == "Linear":
        return load_linear_interpolation(args, reg_eval)
    elif args.method == "SLERP":
        return load_spherical_interpolation(args, reg_eval)
    elif "Fisher-Rao" in args.method:
        return load_fisher_rao(args, reg_eval)
    elif "Jacobian-Metric" in args.method:
        return load_jacobian_met(args, reg_eval)
    elif "Inverse-Density" == args.method:
        return load_inverse_met(args, reg_eval)
    elif "Generative-Metric" == args.method:
        return load_generative_met(args, reg_eval)
    elif "Monge-Metric" == args.method:
        return load_monge_met(args, reg_eval)
    else:
        raise ValueError(f"Invalid method: {args.method}")
        
        
#%% Evaluate method

def evaluate_bvp_method(bvp_geodesic, 
                        reg_eval,
                        x1, 
                        x2,
                        args,
                        ):
    
    curve_method = {}
    curve = bvp_geodesic(x1,x2)

    curve_method['NLL'] = reg_eval(curve).item()
    curve_method['Energy'] = torch.sum((curve[1:]-curve[:-1])**2).item()
    if hasattr(bvp_geodesic, 'lam_norm'):
        curve_method['Reg Energy'] = curve_method['Energy'] + bvp_geodesic.lam_norm*curve_method['NLL']
        curve_method['lam_norm'] = bvp_geodesic.lam_norm
    
    def run_once():
        # For GPU timing, ensure accurate measurement
        if data_sample.is_cuda:
            torch.cuda.synchronize()
        out = bvp_geodesic(x1, x2)
        if data_sample.is_cuda:
            torch.cuda.synchronize()

    timing = timeit.repeat(
        stmt=run_once,
        number=args.number_repeats,
        repeat=args.timing_repeats
    )

    timing = torch.tensor(timing, device=data_sample.device)
    
    curve_method['mean_time'] = torch.mean(timing)
    curve_method['std_time'] = torch.std(timing)
    
    return curve_method

def evaluate_ivp_method(ivp_geodesic, 
                        reg_eval,
                        x1, 
                        args,
                        ):
    
    curve_method = {}
    
    theta = torch.linspace(0, 2*torch.pi,10, device=args.device)
    v = torch.stack([torch.cos(theta), torch.sin(theta)]).T
    
    curve = torch.stack([ivp_geodesic(x1, v0) for v0 in v])

    curve_method['NLL'] = reg_eval(curve).item()
    curve_method['Energy'] = torch.sum((curve[:,1:]-curve[:,:-1])**2).item()
    
    def run_once():
        # For GPU timing, ensure accurate measurement
        if data_sample.is_cuda:
            torch.cuda.synchronize()
        out = ivp_geodesic(x1, v[0])
        if data_sample.is_cuda:
            torch.cuda.synchronize()

    timing = timeit.repeat(
        stmt=run_once,
        number=args.number_repeats,
        repeat=args.timing_repeats
    )

    timing = torch.tensor(timing, device=data_sample.device)
    
    curve_method['mean_time'] = torch.mean(timing)
    curve_method['std_time'] = torch.std(timing)
    
    return curve_method

def evaluate_mean_method(mean_com, 
                         reg_eval,
                         data_sample, 
                         args,
                         ):
    
    curve_method = {}
    
    mean, curve = mean_com(data_sample)

    curve_method['NLL'] = reg_eval(curve).item()
    curve_method['Energy'] = torch.sum((curve[:,1:]-curve[:,:-1])**2).item()
    
    def run_once():
        # For GPU timing, ensure accurate measurement
        if data_sample.is_cuda:
            torch.cuda.synchronize()
        out = mean_com(data_sample)
        if data_sample.is_cuda:
            torch.cuda.synchronize()

    timing = timeit.repeat(
        stmt=run_once,
        number=args.number_repeats,
        repeat=args.timing_repeats
    )

    timing = torch.tensor(timing, device=data_sample.device)
    
    curve_method['mean_time'] = torch.mean(timing).item()
    curve_method['std_time'] = torch.std(timing).item()
    
    return curve_method

def evaluate_method(args, ivp_geodesic, bvp_geodesic, mean_com, reg_eval, x1, x2, data_sample):
    
    if args.computation == "bvp":
        return evaluate_bvp_method(bvp_geodesic, reg_eval, x1, x2, args)
    elif args.computation == "ivp":
        return evaluate_ivp_method(ivp_geodesic, reg_eval, x1, args)
    elif args.computation == "mean":
        return evaluate_mean_method(mean_com, reg_eval, data_sample, args)
    else:
        raise ValueError(f"Invalid computational method: {args.computation}")
        
#%% Compute energy at grid

def compute_grid_energy(
    args,
    data_sample,
    grid,
    reg_eval,
    grid_batch_size=1000,
    ):
    bvp_geodesic = ProbGEORCE_Euclidean_Adaptive_FixedLam(
        reg_fun=reg_eval,
        init_fun=None,
        lam=args.lam,
        N=args.N_grid,
        tol=args.tol,
        max_iter=args.max_iter,
        lr_rate=0.001,
        beta1=0.5,
        beta2=0.5,
        eps=1e-8,
        device=args.device,
    )

    data_sample = data_sample.to(args.device)
    grid = grid.to(args.device)

    num_grid = grid.shape[0]
    num_data = data_sample.shape[0]

    # Accumulator for grid energies
    grid_energy_sum = torch.zeros(num_grid, device=args.device)

    for i, data in enumerate(data_sample):
        print(f"Data computation: {i + 1}/{num_data}")

        # Process grid in batches
        for start in range(0, num_grid, grid_batch_size):
            end = min(start + grid_batch_size, num_grid)
            grid_batch = grid[start:end]

            # Expand data to match batch size
            data_expanded = data.unsqueeze(0).expand(end - start, -1)

            curves = bvp_geodesic(data_expanded, grid_batch)

            # Path energy
            path_energy = torch.sum(
                (curves[:, :, 1:] - curves[:, :, :-1]) ** 2,
                dim=(1, 2),
            )

            # Regularization energy (cannot easily vectorize if reg_eval is Python)
            reg_energy = torch.tensor(
                [reg_eval(curve).item() for curve in curves],
                device=args.device,
            )

            energy = path_energy + args.lam * reg_energy

            grid_energy_sum[start:end] += energy

            # Optional: free memory early
            del curves, path_energy, reg_energy, energy

    # Mean over data samples
    grid_energy = grid_energy_sum / num_data

    return grid_energy


#%% main

if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    #args.tol = args.tol*args.lam
    
    save_path = ''.join((args.save_path, args.runtime_type, '/'))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    if args.runtime_type == "metrics":
            
        save_path = ''.join((save_path, f"{args.model_type}_{args.computation}_{args.method}_{args.geodesic_method}.pkl"))
        
        reg_eval, x1, x2, data_sample, grid = load_model(args)
        
        ivp_geodesic, bvp_geodesic, mean_com = load_method(args, reg_eval)
        
        result = evaluate_method(args, ivp_geodesic, bvp_geodesic, mean_com, reg_eval, x1, x2, data_sample)
    
        print(result)
        
        # Save to pickle
        with open(save_path, "wb") as f:  # wb = write binary
            pickle.dump(result, f)
            
    elif args.runtime_type == "runtime":
        
        lam_str = str(args.lam).replace('.', 'd')
        save_path = ''.join((save_path, f"{args.model_type}_{args.computation}_{args.method}_{lam_str}_{args.geodesic_method}.pkl"))
        
        reg_eval, x1, x2, data_sample, grid = load_model(args)
        
        ivp_geodesic, bvp_geodesic, mean_com = load_method(args, reg_eval)
        
        result = evaluate_method(args, ivp_geodesic, bvp_geodesic, mean_com, reg_eval, x1, x2, data_sample)
    
        print(result)
        
        # Save to pickle
        with open(save_path, "wb") as f:  # wb = write binary
            pickle.dump(result, f)
            
    elif args.runtime_type == "grid":
            
        lam_str = str(args.lam).replace('.', 'd')
        save_path = ''.join((save_path, f"{args.model_type}_{lam_str}.pkl"))
        
        reg_eval, x1, x2, data_sample, grid = load_model(args)
        
        grid_energy = compute_grid_energy(args, data_sample, grid, reg_eval)
        
        result = {'grid': grid,
                  'data_sample': data_sample,
                  'grid_energy': grid_energy,
                  }
        
        with open(save_path, "wb") as f:  # wb = write binary
            pickle.dump(result, f)
    else:
        raise ValueError(f"Invalid runtime_type: {args.runtime_type}")
