#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 09:54:30 2024

@author: fmry
"""

#%% Sources

#%% Modules

from jax_geometry.setup import *

#%% Riemannian Manifold

class RiemannianManifold(ABC):
    def __init__(self,
                 G:Callable[[Array], Array]=None,
                 f:Callable[[Array], Array]=None,
                 invf:Callable[[Array],Array]=None,
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
    
    def Jf(self,
           z:Array
           )->Array:
        
        if self.f is None:
            raise ValueError("Both the pull-back map is not defined")
        else:
            return jacfwd(self.f)(z)
        
    def pull_back_metric(self,
                         z:Array
                         )->Array:
        
        if self.f is None:
            raise ValueError("Both the pull-back map is not defined")
        else:
            Jf = self.Jf(z)
            return jnp.einsum('ik,il->kl', Jf, Jf)
    
    def DG(self,
           z:Array
           )->Array:

        return jacfwd(self.G)(z)
    
    def inner_product(self,
                      z:Array,
                      u:Array,
                      )->Array:
        
        G = self.G(z)
        
        return jnp.einsum('i,ij,j->', u,G,u)
    
    def Ginv(self,
             z:Array
             )->Array:
        
        return jnp.linalg.inv(self.G(z))
    
    def christoffel_symbols(self,
                            z:Array
                            )->Array:
        
        Dgx = self.DG(z)
        gsharpx = self.Ginv(z)
        
        return 0.5*(jnp.einsum('im,kml->ikl',gsharpx,Dgx)
                   +jnp.einsum('im,lmk->ikl',gsharpx,Dgx)
                   -jnp.einsum('im,klm->ikl',gsharpx,Dgx))
    
    def geodesic_equation(self,
                          z:Array,
                          v:Array
                          )->Array:
        
        Gamma = self.christoffel_symbols(z)

        dx1t = v
        dx2t = -jnp.einsum('ikl,k,l->i',Gamma,v,v)
        
        return jnp.hstack((dx1t,dx2t))
    
    def Exp_ode(self,
                z:Array,
                v:Array,
                T:int=100,
                )->Array:
        
        def dif_fun(t,y):
            
            z = y[:self.dim]
            v = y[self.dim:]
            
            return self.geodesic_equation(z, v)
        
        dim = len(z)
        
        ts, zs = rk45_fixed_step(dif_fun, 
                                 jnp.hstack((z, v)), 
                                 0.0, 
                                 1.0,
                                 T,
                                 )
        
        zs = zs[:,:self.dim]
        
        return zs
    
    def Exp_ode(self,
                z:Array,
                v:Array,
                T:int=100,
                )->Array:
        
        def dif_fun(t,y):
            
            z = y[:self.dim]
            v = y[self.dim:]
            
            return self.geodesic_equation(z, v)
        
        dim = len(z)
        
        ts, zs = rk45_fixed_step(dif_fun, 
                                 jnp.hstack((z, v)), 
                                 0.0, 
                                 1.0,
                                 T,
                                 )
        
        zs = zs[:,:self.dim]
        
        return zs
    
    def parallel_transport_ode(self,
                               gamma:Array,
                               v:Array,
                               )->Array:

        """
        Parallel transport vector v0 along a curve gamma using RK45 (scipy).
        
        Parameters:
            gamma_points: (N, n) array of curve points
            t_vals: (N,) array of time parameters (monotonic)
            v0: (n,) initial vector at gamma[0]
            compute_christoffel: function: point -> (n, n, n)
        
        Returns:
            t_vals: same as input
            transported_vs: (N, n) transported vectors
        """
        
        T = gamma.shape[0]
        t_vals = jnp.linspace(0.,1.,T, endpoint=True) / T
    
        # Step 1: Create cubic interpolators for gamma and its derivative
        gamma_spline = CurveInterpolator(gamma, t_vals)
    
        # Step 2: Define the ODE for dV/dt = -Γ^i_{jk} * dx^j/dt * V^k
        def odefun(t, V):
            p, dxdt = gamma_spline(t)
            Gamma = self.christoffel_symbols(p)
            contraction = jnp.einsum('ijk,j,k->i', Gamma, dxdt, V)
            return -contraction
        
        
        ts, vs = rk45_fixed_step(odefun, 
                                 v, 
                                 0.0, 
                                 1.0,
                                 T,
                                 )
        
        return ts, vs

    def energy(self, 
               gamma:Array,
               )->Array:
        
        T = len(gamma)-1
        dt = 1.0/T
        dgamma = (gamma[1:]-gamma[:-1])*T
        
        g = vmap(lambda g: self.G(g))(gamma)
        integrand = jnp.einsum('ti,tij,tj->t', dgamma, g[:-1], dgamma)
        
        return jnp.trapezoid(integrand, dx=dt)
    
    def length(self,
               gamma:Array,
               )->Array:
        
        T = len(gamma)-1
        dt = 1.0/T
        
        dgamma = (gamma[1:]-gamma[:-1])*T
        
        g = vmap(lambda g: self.G(g))(gamma)
        integrand = jnp.sqrt(jnp.einsum('ti,tij,tj->t', dgamma, g[:-1], dgamma))
            
        return jnp.trapezoid(integrand, dx=dt)
    
    def length_frechet(self, 
                       zt:Array,
                       z_obs:Array,
                       z_mu:Array,
                       )->Array:
        
        def step_length(length:Array,
                         y:Tuple,
                         )->Tuple:
            
            z0, z_path = y
            
            length += self.path_length_frechet(z0, z_path, z_mu, G0)**2
            
            return (length,)*2
        
        G0 = self.G(z_mu)
        length, _ = lax.scan(step_length,
                             init=0.0,
                             xs=(z_obs, zt),
                             )
        
        return length
    
    def path_length_frechet(self, 
                            zT:Array,
                            zt:Array,
                            mu:Array,
                            G0:Array,
                            )->Array:
        
        term1 = zt[0]-mu
        val1 = jnp.sqrt(jnp.einsum('i,ij,j->', term1, G0, term1))
        
        term2 = zt[1:]-zt[:-1]
        Gt = vmap(lambda z: self.G(z))(zt)
        val2 = jnp.sqrt(jnp.einsum('ti,tij,tj->t', term2, Gt[:-1], term2))
        
        term3 = zT-zt[-1]
        val3 = jnp.sqrt(jnp.einsum('i,ij,j->', term3, Gt[-1], term3))
        
        return val1+jnp.sum(val2)+val3
    
    def indicatrix(self,
                   z:Array,
                   N_points:int=100,
                   *args,
                   )->Array:
        
        theta = jnp.linspace(0.,2*jnp.pi,N_points)
        u = jnp.vstack((jnp.cos(theta), jnp.sin(theta))).T
        
        norm = vmap(self.inner_product, in_axes=(None, 0))(z,u)
        
        return jnp.einsum('ij,i->ij', u, 1./norm)
    
#%% Lambda Manifold

class LambdaManifold(ABC):
    def __init__(self,
                 M:RiemannianManifold,
                 S:Callable[[Array], Array],
                 lam:float=1.0,
                 )->None:
        
        self.M = M
        self.S = S
        self.lam = lam
            
        return
        
    def __str__(self)->str:
        
        return "Lambda Manifold base object"
    
    def inner_product(self,
                      z:Array,
                      u:Array,
                      )->Array:
        
        G = self.M.G(z)
        
        return jnp.einsum('i,ij,j->', u,G,u) + self.lam*self.S(z)
    
    def Ginv(self,
             z:Array
             )->Array:
        
        return jnp.linalg.inv(self.M.G(z))
    
    def christoffel_symbols(self,
                            z:Array
                            )->Array:
        
        return self.M.christoffel_symbols(z)
        
    def geodesic_equation(self,
                          z:Array,
                          v:Array
                          )->Array:
        
        Gamma = self.M.christoffel_symbols(z)
        grad_s = grad(self.S)(z)
        Ginv = self.M.Ginv(z)

        dx1t = v
        dx2t = (
            -jnp.einsum('ikl,k,l->i',Gamma,v,v) - 0.5*self.lam*jnp.einsum('ij,j->i', Ginv, grad_s)
            )
        
        return jnp.hstack((dx1t,dx2t))
    
    def Exp(self,
            z:Array,
            v:Array,
            T:int=100,
            )->Array:
        
        def dif_fun(t,y):
            
            z = y[:self.M.dim]
            v = y[self.M.dim:]
            
            return self.geodesic_equation(z, v)
        
        dim = len(z)
        
        ts, zs = rk45_fixed_step(dif_fun, 
                                 jnp.hstack((z, v)), 
                                 0.0, 
                                 1.0,
                                 T,
                                 )
        
        zs = zs[:,:self.dim]
        
        return zs
    
    def Exp_ode(self,
                z:Array,
                v:Array,
                T:int=100,
                )->Array:
        
        def dif_fun(t,y):
            
            z = y[:self.M.dim]
            v = y[self.M.dim:]
            
            return self.geodesic_equation(z, v)
        
        dim = len(z)
        
        ts, zs = rk45_fixed_step(dif_fun, 
                                 jnp.hstack((z, v)), 
                                 0.0, 
                                 1.0,
                                 T,
                                 )
        
        zs = zs[:,:self.M.dim]
        
        return zs
    
    def parallel_transport_ode(self,
                               gamma:Array,
                               v:Array,
                               )->Array:

        """
        Parallel transport vector v0 along a curve gamma using RK45 (scipy).
        
        Parameters:
            gamma_points: (N, n) array of curve points
            t_vals: (N,) array of time parameters (monotonic)
            v0: (n,) initial vector at gamma[0]
            compute_christoffel: function: point -> (n, n, n)
        
        Returns:
            t_vals: same as input
            transported_vs: (N, n) transported vectors
        """
        
        T = gamma.shape[0]
        t_vals = jnp.linspace(0.,1.,T, endpoint=True) / T
    
        # Step 1: Create cubic interpolators for gamma and its derivative
        gamma_spline = CurveInterpolator(gamma, t_vals)
    
        # Step 2: Define the ODE for dV/dt = -Γ^i_{jk} * dx^j/dt * V^k
        def odefun(t, V):
            p, dxdt = gamma_spline(t)
            Gamma = self.christoffel_symbols(p)
            contraction = jnp.einsum('ijk,j,k->i', Gamma, dxdt, V)
            return -contraction
        
        
        ts, vs = rk45_fixed_step(odefun, 
                                 v, 
                                 0.0, 
                                 1.0,
                                 T,
                                 )
        
        return ts, vs

    def energy(self, 
               gamma:Array,
               )->Array:
        
        T = len(gamma)-1
        dt = 1.0/T
        dgamma = (gamma[1:]-gamma[:-1])*T
        
        g = vmap(lambda g: self.M.G(g))(gamma)
        integrand = jnp.einsum('ti,tij,tj->t', dgamma, g[:-1], dgamma)
        integrand += self.lam*vmap(self.S)(gamma)
        
        return jnp.trapezoid(integrand, dx=dt)
    
    def length(self,
               gamma:Array,
               )->Array:
        
        T = len(gamma)-1
        dt = 1.0/T
        
        dgamma = (gamma[1:]-gamma[:-1])*T
        
        g = vmap(lambda g: self.M.G(g))(gamma)
        integrand = jnp.sqrt(jnp.einsum('ti,tij,tj->t', dgamma, g[:-1], dgamma) + self.lam*vmap(self.S)(gamma))
            
        return jnp.trapezoid(integrand, dx=dt)
    
    def length_frechet(self, 
                       zt:Array,
                       z_obs:Array,
                       z_mu:Array,
                       )->Array:
        
        def step_length(length:Array,
                         y:Tuple,
                         )->Tuple:
            
            z0, z_path = y
            
            length += self.path_length_frechet(z0, z_path, z_mu, G0)**2
            
            return (length,)*2
        
        G0 = self.M.G(z_mu)
        length, _ = lax.scan(step_length,
                             init=0.0,
                             xs=(z_obs, zt),
                             )
        
        return length
    
    def path_length_frechet(self, 
                            zT:Array,
                            zt:Array,
                            mu:Array,
                            G0:Array,
                            )->Array:
        
        term1 = zt[0]-mu
        val1 = jnp.sqrt(jnp.einsum('i,ij,j->', term1, G0, term1) + self.lam*self.S(mu))
        
        term2 = zt[1:]-zt[:-1]
        Gt = vmap(lambda z: self.M.G(z))(zt)
        S_val = vmap(self.S)(zt)
        val2 = jnp.sqrt(jnp.einsum('ti,tij,tj->t', term2, Gt[:-1], term2) + self.lam*S_val[:-1])
        
        term3 = zT-zt[-1]
        val3 = jnp.sqrt(jnp.einsum('i,ij,j->', term3, Gt[-1], term3) + self.lam*S_val[-1])
        
        return val1+jnp.sum(val2)+val3
    
    def indicatrix(self,
                   z:Array,
                   N_points:int=100,
                   *args,
                   )->Array:
        
        theta = jnp.linspace(0.,2*jnp.pi,N_points)
        u = jnp.vstack((jnp.cos(theta), jnp.sin(theta))).T
        
        norm = vmap(self.inner_product, in_axes=(None, 0))(z,u)
        
        return jnp.einsum('ij,i->ij', u, 1./norm)

#%% ODE Fun

def rk45_step(f, t, y, dt):
    """Single Dormand–Prince RK45 step."""
    # RK45 coefficients
    c = jnp.array([0.0, 1/5, 3/10, 4/5, 8/9, 1.0, 1.0])
    
    a = [
        [],
        [1/5],
        [3/40, 9/40],
        [44/45, -56/15, 32/9],
        [19372/6561, -25360/2187, 64448/6561, -212/729],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656],
        [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84],
    ]
    
    b5 = jnp.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0])  # 5th order
    b4 = jnp.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])  # 4th order

    k = []

    for i in range(7):
        dy = sum(a[i][j] * k[j] for j in range(i)) if i > 0 else 0.0
        k_i = f(t + c[i] * dt, y + dt * dy)
        k.append(k_i)

    k = jnp.stack(k)  # shape (7, ...) for broadcasting

    y_next = y + dt * jnp.tensordot(b5, k, axes=1)
    y_err = dt * jnp.tensordot(b5 - b4, k, axes=1)

    return y_next, y_err

    
def rk45_fixed_step(f, y0, t0, t1, T):
    """Fixed-step RK45 ODE solver."""

    ts = jnp.linspace(t0, t1, T + 1)
    dt = 1./T

    def step_fn(y, t):
        y_next, _ = rk45_step(f, t, y, dt)
        return y_next, y_next

    _, ys = lax.scan(step_fn, y0, ts[:-1])  # exclude last t since we return y0 + N steps
    ys = jnp.vstack([y0[None, :], ys])      # prepend initial condition

    return ts, ys

#%% Spline fun

class CurveInterpolator:
    def __init__(self, points: jnp.ndarray, t_domain: jnp.ndarray = None):
        """
        Initialize with:
            - points: array of shape (N, d), N curve points in d dimensions
            - t_domain: array of shape (N,), optional time parameter values (defaults to linspace [0,1])
        """
        self.points = points
        self.N, self.d = points.shape

        if t_domain is None:
            self.t_domain = jnp.linspace(0.0, 1.0, self.N)
        else:
            assert t_domain.shape[0] == self.N
            self.t_domain = t_domain

        self.tangents = self._compute_tangents(self.points)

    def _compute_tangents(self, y):
        """Estimate tangents using central differences."""
        dy = jnp.zeros_like(y)
        dy = dy.at[1:-1].set((y[2:] - y[:-2]) / 2.0)
        dy = dy.at[0].set(y[1] - y[0])
        dy = dy.at[-1].set(y[-1] - y[-2])
        return dy

    def _hermite_basis(self, u):
        """Cubic Hermite basis and their derivatives."""
        h00 = 2*u**3 - 3*u**2 + 1
        h10 = u**3 - 2*u**2 + u
        h01 = -2*u**3 + 3*u**2
        h11 = u**3 - u**2

        dh00 = 6*u**2 - 6*u
        dh10 = 3*u**2 - 4*u + 1
        dh01 = -6*u**2 + 6*u
        dh11 = 3*u**2 - 2*u

        return (h00, h10, h01, h11), (dh00, dh10, dh01, dh11)

    def _interp_single(self, t):
        """Interpolate a single t value."""
        t = jnp.clip(t, self.t_domain[0], self.t_domain[-1])
        i = jnp.searchsorted(self.t_domain, t, side='right') - 1
        i = jnp.clip(i, 0, self.N - 2)

        t0 = self.t_domain[i]
        t1 = self.t_domain[i + 1]
        dt = t1 - t0
        u = (t - t0) / dt

        p0 = self.points[i]
        p1 = self.points[i + 1]
        m0 = self.tangents[i]
        m1 = self.tangents[i + 1]

        (h00, h10, h01, h11), (dh00, dh10, dh01, dh11) = self._hermite_basis(u)

        value = (
            h00 * p0 +
            h10 * dt * m0 +
            h01 * p1 +
            h11 * dt * m1
        )

        derivative = (
            dh00 * p0 +
            dh10 * dt * m0 +
            dh01 * p1 +
            dh11 * dt * m1
        ) / dt  # Chain rule

        return value, derivative

    def __call__(self, t):
        """
        Evaluate the interpolated curve and its derivative at t.
        t: scalar or array in [t0, tN]
        Returns: (curve(t), curve'(t))
        """
        if jnp.ndim(t) == 0:
            return self._interp_single(t)
        else:
            return jax.vmap(self._interp_single)(t)
    