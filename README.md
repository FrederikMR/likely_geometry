# A Likely Geometry of Generative Models.

Experimential results on ControlNet for method described in https://github.com/FrederikMR/likely_geometry.

<p align="center">
  <img src="https://github.com/user-attachments/files/25181960/pgeorce_noise_mean_grid_left_mod.pdf" width="700" />
</p>



## Installation and Requirements

The implementations in the GitHub is Python 3.10.12 and has been implemented in both JAX and PyTorch.


## Reproducing Experiments

All experiments can be re-produced by running the notebooks and the runtime.py and run_gen2d_geometry for the given data, method and hyper-parameters. The ControlNet interpolation can be found at https://github.com/FrederikMR/controlnet_interpolation/tree/main. The Energy-based interpolation can be found at https://github.com/FrederikMR/ebm_interpolation.

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


