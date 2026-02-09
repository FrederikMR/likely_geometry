# A Likely Geometry of Generative Models
This GitHub contains the code for the paper "A Likely Geometry of Generative Models" that considers interpolation and other statistics on generative models as a Newtonian system on a Riemannian manifold.

<p align="center">
  <img src="https://github.com/user-attachments/files/25181487/pga_mod2_wide.pdf" width="800" />
</p>

## Installation and Requirements

The implementations in the GitHub is Python 3.10.12 and has been implemented in both JAX and PyTorch.

## Application to diffusion models

In the paper, we apply our method to ControlNet. The code for this can be found at:

https://github.com/FrederikMR/controlnet_interpolation

## Code Structure

The following shows the structure of the code. All general implementations of geometry and optimization algorithms can be found in the "torch_geometry" or "jax_geometry" folder for both the Riemannian and Finsler case. Note that they differ a bit in terms of what manifolds are defined within them. The selected file structure is the following

    .
    ├── load_manifold.py                   # Load manifolds and points for connecting geodesic
    ├── runtime.py                         # Times length and runtime for different optimization algorithms to consturct interpolation
    ├── run_gen2d_models.py                # Computes runtime and estimates for the trained 2d generative models
    ├── train_gen2d_models.py              # Training EBM, NF, AR and VAE model.
    ├── gen2d_models.ipynb                 # Plots the results on the 2d generative models
    ├── ddpm2d.ipynb                       # Train and compute interpolation on a simple dinasour model (see github.com/tanelp/tiny-diffusion)
    └── README.md

## Reproducing Experiments

All experiments can be re-produced by running the notebooks and the runtime.py and run_gen2d_models.py package for the given manifold, hyper-parameters and optimization method.

## Reference

If you want to use the algorithm or the method proposed in the paper for scientific purposes, please cite:


    @misc{rygaard2026likelygeometrygenerativemodels,
          title={A Likely Geometry of Generative Models}, 
          author={Frederik Möbius Rygaard and Shen Zhu and Yinzhu Jin and Søren Hauberg and Tom Fletcher},
          year={2026},
          eprint={2510.26266},
          archivePrefix={arXiv},
          primaryClass={cs.LG},
          url={https://arxiv.org/abs/2510.26266}, 
    }
