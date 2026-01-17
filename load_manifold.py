#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 09:47:42 2024

@author: fmry
"""

#%% Sources

#%% Modules

import jax.random as jrandom
import jax.numpy as jnp
import jax.scipy as jscipy

from jax import jit, lax, vmap

import haiku as hk

from jax_geometry.manifolds import nSphere, nEllipsoid, nEuclidean, \
    nParaboloid, HyperbolicParaboloid, SPDN, T2, LatentSpaceManifold, FisherRaoGeometry

#%% Load manifolds

def load_manifold(manifold:str="Euclidean", 
                  dim:int = 2,
                  svhn_path:str = "../../../Data/SVHN/",
                  celeba_path:str = "../../../Data/CelebA/",
                  ):
    
    rho = 0.5 #default
    
    key = jrandom.key(2712)
    key, subkey = jrandom.split(key)
    
    n_clusters = 3
    weights = jrandom.uniform(subkey, shape=(n_clusters,))
    weights /= jnp.sum(weights)
    
    if manifold == "Euclidean":
        M = nEuclidean(dim=dim)
        z0 = -jnp.linspace(0,1,dim)
        zT = jnp.ones(dim, dtype=jnp.float32)
        rho = 0.5
        
        key, subkey = jrandom.split(key)
        centroids = z0 + 1.0*jrandom.normal(key, shape=(n_clusters, len(z0)))
        reg_fun = lambda x: -jnp.sum(vmap(lambda w, mean,cov: w*jscipy.stats.multivariate_normal.logpdf(x,
                                                                                                        mean=centroids, 
                                                                                                        cov=jnp.eye(len(z0)),
                                                                                                        ))(weights, centroids),
                                     )
    if manifold == "SPDN":
        M = SPDN(N=dim)
        x0 = jnp.eye(dim)
        
        z0 = M.invf(x0)
        zT = jnp.linspace(0.5,1.0, M.dim)
        rho = 0.5
        key, subkey = jrandom.split(key)
        centroids = z0 + 1.0*jrandom.normal(key, shape=(n_clusters, len(z0)))
        
        reg_fun = lambda x: -jnp.sum(vmap(lambda w, mean: w*jscipy.stats.multivariate_normal.logpdf(x,
                                                                                                     mean=mean, 
                                                                                                     cov=jnp.eye(len(z0)),
                                                                                                     ))(weights, centroids),
                                     )
    elif manifold == "Paraboloid":
        M = nParaboloid(dim=dim)
        z0 = -jnp.linspace(0,1,dim)
        zT = jnp.ones(dim, dtype=jnp.float32)
        rho = 0.5
        key, subkey = jrandom.split(key)
        centroids = z0 + 1.0*jrandom.normal(key, shape=(n_clusters, len(z0)))
        reg_fun = lambda x: -jnp.sum(vmap(lambda w, mean: w*jscipy.stats.multivariate_normal.logpdf(x,
                                                                                                     mean=mean, 
                                                                                                     cov=jnp.eye(len(z0)),
                                                                                                     ))(weights, centroids),
                                     )
    elif manifold == "Sphere":
        M = nSphere(dim=dim)
        z0 = -jnp.linspace(0,1,dim)
        zT = 0.5*jnp.ones(dim, dtype=jnp.float32)
        rho = .5
        key, subkey = jrandom.split(key)
        centroids = z0 + 1.0*jrandom.normal(key, shape=(n_clusters, len(z0)))
        reg_fun = lambda x: -jnp.sum(vmap(lambda x: vmap(lambda w, mean: w*jscipy.stats.multivariate_normal.logpdf(x,
                                                                                                     mean=mean, 
                                                                                                     cov=jnp.eye(len(z0)),
                                                                                                     ))(weights, centroids),
                                     )(x.reshape(-1,len(z0))))
    elif manifold == "Ellipsoid":
        params = jnp.linspace(0.5,1.0,dim+1)
        M = nEllipsoid(dim=dim, params=params)
        z0 = -jnp.linspace(0,1,dim)
        zT = 0.5*jnp.ones(dim, dtype=jnp.float32)
        rho = 0.5
        key, subkey = jrandom.split(key)
        centroids = z0 + 1.0*jrandom.normal(key, shape=(n_clusters, len(z0)))
        reg_fun = lambda x: -jnp.sum(vmap(lambda w, mean: w*jscipy.stats.multivariate_normal.logpdf(x,
                                                                                                     mean=mean, 
                                                                                                     cov=jnp.eye(len(z0)),
                                                                                                     ))(weights, centroids),
                                     )
    elif manifold == "T2":
        M = T2(R=3.0, r=1.0)
        z0 = jnp.array([0.0, 0.0])
        zT = jnp.array([5*jnp.pi/4, 5*jnp.pi/4])
        rho = 0.5
        key, subkey = jrandom.split(key)
        centroids = z0 + 1.0*jrandom.normal(key, shape=(n_clusters, len(z0)))
        reg_fun = lambda x: -jnp.sum(vmap(lambda w, mean: w*jscipy.stats.multivariate_normal.logpdf(x,
                                                                                                     mean=mean, 
                                                                                                     cov=jnp.eye(len(z0)),
                                                                                                     ))(weights, centroids),
                                     )
    elif manifold == "Gaussian":
        M = FisherRaoGeometry(distribution='Gaussian')
        z0 = jnp.array([-1.0, 0.5])
        zT = jnp.array([1.0, 1.0])
        rho = .5
        key, subkey = jrandom.split(key)
        centroids = z0 + 1.0*jrandom.normal(key, shape=(n_clusters, len(z0)))
        reg_fun = lambda x: -jnp.sum(vmap(lambda w, mean: w*jscipy.stats.multivariate_normal.logpdf(x,
                                                                                                     mean=mean, 
                                                                                                     cov=jnp.eye(len(z0)),
                                                                                                     ))(weights, centroids),
                                     )
    elif manifold == "Frechet":
        M = FisherRaoGeometry(distribution='Frechet')
        z0 = jnp.array([0.5, 0.5])
        zT = jnp.array([1.0, 1.0])
        rho = 0.5
        key, subkey = jrandom.split(key)
        centroids = z0 + 1.0*jrandom.normal(key, shape=(n_clusters, len(z0)))
        reg_fun = lambda x: -jnp.sum(vmap(lambda w, mean: w*jscipy.stats.multivariate_normal.logpdf(x,
                                                                                                     mean=mean, 
                                                                                                     cov=jnp.eye(len(z0)),
                                                                                                     ))(weights, centroids),
                                     )
    elif manifold == "Cauchy":
        M = FisherRaoGeometry(distribution='Cauchy')
        z0 = jnp.array([-1.0, 0.5])
        zT = jnp.array([1.0, 1.0])
        rho = 0.5
        key, subkey = jrandom.split(key)
        centroids = z0 + 1.0*jrandom.normal(key, shape=(n_clusters, len(z0)))
        reg_fun = lambda x: -jnp.sum(vmap(lambda w, mean: w*jscipy.stats.multivariate_normal.logpdf(x,
                                                                                                     mean=mean, 
                                                                                                     cov=jnp.eye(len(z0)),
                                                                                                     ))(weights, centroids),
                                     )
    elif manifold == "Pareto":
        M = FisherRaoGeometry(distribution='Pareto')
        z0 = jnp.array([0.5, 0.5])
        zT = jnp.array([1.0, 1.0])
        rho = 0.5
        key, subkey = jrandom.split(key)
        centroids = z0 + 1.0*jrandom.normal(key, shape=(n_clusters, len(z0)))
        reg_fun = lambda x: -jnp.sum(vmap(lambda w, mean: w*jscipy.stats.multivariate_normal.logpdf(x,
                                                                                                     mean=mean, 
                                                                                                     cov=jnp.eye(len(z0)),
                                                                                                     ))(weights, centroids),
                                     )
    else:
        raise ValueError(f"Manifold, {manifold}, is not defined. Only suported is: \n\t-Euclidean\n\t-Paraboloid\n\t-Sphere")
        
    return z0, zT, M, reg_fun, rho