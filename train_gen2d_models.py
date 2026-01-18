#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 12:30:28 2025

@author: fmry
"""

#%% modules

# Import required packages
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np

import normflows as nf

from torchebm.core import BaseEnergyFunction
from torchebm.samplers import LangevinDynamics
from torchebm.losses import ContrastiveDivergence
from torchebm.datasets import CheckerboardDataset #, GaussianMixtureDataset, PinwheelDataset

#argparse
import argparse

from tqdm import tqdm

from typing import List, Any

#%% Args Parser

def parse_args():
    parser = argparse.ArgumentParser()
    # File-paths
    parser.add_argument('--gen_model', default="vae",
                        type=str)
    parser.add_argument('--device', default="cpu",
                        type=str)
    parser.add_argument('--save_path', default='../models/gen2d/',
                        type=str)

    args = parser.parse_args()
    return args

#%% Train normalizing flows

def train_nf(args):
    
    # Set up model

    # Define 2D Gaussian base distribution
    base = nf.distributions.base.DiagGaussian(2)
    
    # Define list of flows
    num_layers = 32
    flows = []
    for i in range(num_layers):
        # Neural network with two hidden layers having 64 units each
        # Last layer is initialized by zeros making training more stable
        param_map = nf.nets.MLP([1, 64, 64, 2], init_zeros=True)
        # Add flow layer
        flows.append(nf.flows.AffineCouplingBlock(param_map))
        # Swap dimensions
        flows.append(nf.flows.Permute(2, mode='swap'))
        
    # Construct flow model
    model = nf.NormalizingFlow(base, flows)
    
    # Move model on GPU if available
    device = args.device
    model = model.to(device)
    
    # Define target distribution
    target = nf.distributions.TwoMoons()
    
    # Train model
    max_iter = 4000
    num_samples = 2 ** 9

    loss_hist = np.array([])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
    
    for it in tqdm(range(max_iter)):
        optimizer.zero_grad()
        
        # Get training samples
        x = target.sample(num_samples).to(device)
        
        # Compute loss
        loss = model.forward_kld(x)
        
        # Do backprop and optimizer step
        if ~(torch.isnan(loss) | torch.isinf(loss)):
            loss.backward()
            optimizer.step()
        
        # Log loss
        loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())
        
    # Save the model and the loss history
    save_path = ''.join((args.save_path, 'nf/two_moons.pt'))
    torch.save({
        'model_state_dict': model.state_dict(),
        'loss_history': loss_hist
    }, save_path)
    
    print(f"Model and loss history saved to {save_path}")
    
#%% Save energy model

def train_ebm(args):
    
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
    
    
    # Hyperparameters
    N_SAMPLES = 50_000
    INPUT_DIM = 2
    HIDDEN_DIM = 128
    BATCH_SIZE = 128
    EPOCHS = 1_000
    LEARNING_RATE = 1e-4
    SAMPLER_STEP_SIZE = 0.1
    SAMPLER_NOISE_SCALE = 0.1
    CD_K = 10
    USE_PCD = False
    SEED = 42

    # Device
    device = args.device
    print(f"Using device: {device}")

    # Data Loading
    #dataset = GaussianMixtureDataset(
    #    n_samples=N_SAMPLES,
    #    n_components=4,
    #    std=0.1,
    #    radius=1.5,
    #    device=device,
    #    seed=SEED,
    #)
    #dataset = PinwheelDataset(
    #    n_samples=N_SAMPLES,
    #    n_classes=5,
    #    noise=0.1,
    #    device=device,
    #    seed=SEED,
    #)
    
    dataset = CheckerboardDataset(
        n_samples = N_SAMPLES,
        range_limit = 2.0,
        noise = 0.05,
        device=device,
        seed = SEED,
        )

    # Get the full tensor for visualization purposes
    real_data_for_plotting = dataset.get_data()
    print(f"Data batch_shape: {real_data_for_plotting.shape}")

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )

    # Model Components
    energy_model = MLPEnergy(INPUT_DIM, HIDDEN_DIM).to(device)
    sampler = LangevinDynamics(
        energy_function=energy_model,
        step_size=SAMPLER_STEP_SIZE,
        noise_scale=SAMPLER_NOISE_SCALE,
        device=device,
    )
    loss_fn = ContrastiveDivergence(
        energy_function=energy_model, sampler=sampler, k_steps=CD_K, persistent=USE_PCD
    ).to(device)

    # Optimizer
    optimizer = optim.Adam(energy_model.parameters(), lr=LEARNING_RATE)

    print("Starting training...")
    for epoch in range(EPOCHS):
        energy_model.train()
        epoch_loss = 0.0
        for i, data_batch in enumerate(dataloader):
            optimizer.zero_grad()
            loss, negative_samples = loss_fn(data_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(energy_model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{EPOCHS}], Average Loss: {avg_epoch_loss:.4f}")

    # Save the model and the loss history
    save_path = ''.join((args.save_path, 'ebm/checkboard.pt'))
    torch.save({
        'model_state_dict': energy_model.state_dict(),
    }, save_path)

    print("Training finished.")
    
#%% autoregressive model

def train_ar_model(args):
    
    # ---------------------
    # 1. Toy Dataset
    # ---------------------
    n_samples = 50_000
    x1 = torch.linspace(-3, 3, n_samples).unsqueeze(1)
    x2 = torch.sin(x1) + 0.05 * torch.randn_like(x1)  # sine wave + noise
    x = torch.cat([x1, x2], dim=1)
    
    # ---------------------
    # 2. Autoregressive Model
    # ---------------------
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
    
    # ---------------------
    # 3. Training
    # ---------------------
    model = SimpleAutoregressive2D()
    optimizer = torch.optim.Adam(model.parameters(), lr=51e-4, weight_decay=1e-5)
    
    # Example: wrap x (and optionally y) in a dataset
    dataset = TensorDataset(x)  # or TensorDataset(x, y) if supervised
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True)
    
    epochs=10_000
    for epoch in range(epochs):
        for batch in dataloader:
            xb = batch[0]  # or xb, yb = batch if you have labels
    
            optimizer.zero_grad()
            logp = model(xb)
            loss = -logp.mean()  # same objective
            loss.backward()
            optimizer.step()
    
        print(f"Epoch {epoch}: loss = {loss.item():.3f}")
        
        
    save_path = ''.join((args.save_path, 'ar/sinus.pt'))
    torch.save({
        'model_state_dict': model.state_dict(),
    }, save_path)
    
    return

#%% Train vae

def train_vae(args):

    device = args.device
    
    # ---------------------
    # 1. 3D Circle Dataset (embedded in R^3)
    # ---------------------
    n_samples = 50_000
    epochs=5_000
    theta = torch.linspace(0, 2 * torch.pi, n_samples)
    x1 = torch.cos(theta)
    x2 = torch.sin(theta)
    x3 = torch.zeros_like(theta)  # embed in R^3
    x = torch.stack([x1, x2, x3], dim=1) + 0.1 * torch.randn(n_samples, 3)
    x = x.to(device)
    
    # ---------------------
    # 2. VAE Definition (3D → 2D latent)
    # ---------------------
    #The training script should be modified for the version below.
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
    
        def elbo_latent(self, z, n_samples:int=100):
    
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
            elbo = (kld - rec_loss)
            elbo = elbo.mean()
            
            return z, x_hat, mu, std, kld.mean(), -rec_loss.mean(), elbo
            
        def elbo_latent2(self, z, n_samples: int = 100):
            """
            Evaluate the (approximate) ELBO at a given latent position z.
            This gives an 'energy landscape' over the latent space.
            
            Args:
                z (torch.Tensor): latent coordinates, shape [batch_size, latent_dim]
                n_samples (int): number of samples for Monte Carlo estimation of reconstruction likelihood
        
            Returns:
                elbo_z (torch.Tensor): ELBO value for each latent coordinate
            """
            # Decode z -> reconstructed x_hat
            x_hat = self.decoder(z)
            
            # Reconstruction likelihood under p(x|z)
            log_pxz = self.gaussian_likelihood(x_hat, self.log_scale, x_hat)  # self-likelihood
            
            # Prior p(z) = N(0, I)
            p = torch.distributions.Normal(torch.zeros_like(z), torch.ones_like(z))
            log_pz = p.log_prob(z).sum(-1)
            
            # Approximate posterior q(z|x) ≈ N(z, I * small_var) since we don't have x
            # Here we ignore the encoder distribution and focus on how p(x|z) and p(z) behave
            elbo_z = log_pxz + log_pz  # proportional to log p(x,z)
            
            return elbo_z
    
    
    # ---------------------
    # 3. Training
    # ---------------------
    DATA = x
    
    if device == 'cpu':
        trainloader = DataLoader(dataset = DATA, batch_size= 512,
                                 shuffle = True, pin_memory=True, num_workers = 1)
    else:
        trainloader = DataLoader(dataset = DATA, batch_size= 512,
                                 shuffle = True)
        
    N = len(trainloader.dataset)
    
    model = VAE_3d().to(device) #Model used
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    model.train()
    train_loss_elbo = [] #Elbo loss
    train_loss_rec = [] #Reconstruction loss
    train_loss_kld = [] #KLD loss
    for epoch in range(epochs):
        running_loss_elbo = 0.0
        running_loss_rec = 0.0
        running_loss_kld = 0.0
        for x in trainloader:
            #x = x.to(args.device) #If DATA is not saved to device
            _, x_hat, mu, var, kld, rec_loss, elbo = model(x)
            optimizer.zero_grad() #Based on performance tuning
            elbo.backward()
            optimizer.step()
    
            running_loss_elbo += elbo.item()
            running_loss_rec += rec_loss.item()
            running_loss_kld += kld.item()
    
            #del x, x_hat, mu, var, kld, rec_loss, elbo #In case you run out of memory
    
        train_epoch_loss = running_loss_elbo/N
        train_loss_elbo.append(train_epoch_loss)
        train_loss_rec.append(running_loss_rec/N)
        train_loss_kld.append(running_loss_kld/N)
        print(f"Epoch {epoch+1}/{epochs} - loss: {train_epoch_loss:.4f}")
        
        
    save_path = ''.join((args.save_path, 'vae/circle.pt'))
    torch.save({
        'model_state_dict': model.state_dict(),
    }, save_path)
    

#%% main

if __name__ == '__main__':
    
    args = parse_args()
    
    if args.gen_model == "nf":
        train_nf(args)
    elif args.gen_model == "ebm":
        train_ebm(args)
    elif args.gen_model == "vae":
        train_vae(args)
    elif args.gen_model == "ar":
        train_ar_model(args)

