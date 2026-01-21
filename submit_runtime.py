#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 17:22:08 2024

@author: fmry
"""

#%% Modules

import numpy as np

import os

import time

#%% Submit job

def submit_job():
    
    os.system("bsub < submit_runtime.sh")
    
    return

#%% Generate jobs

def generate_job(manifold, d, T, method, tol, lam):

    with open ('submit_runtime.sh', 'w') as rsh:
        rsh.write(f'''\
    #! /bin/bash
    #BSUB -q gpuv100
    #BSUB -J {method}_{manifold}{d}_{lam}_{T}
    #BSUB -n 4
    #BSUB -gpu "num=1:mode=exclusive_process"
    #BSUB -W 24:00
    #BSUB -R "span[hosts=1]"
    #BSUB -R "rusage[mem=10GB]"
    #BSUB -u fmry@dtu.dk
    #BSUB -B
    #BSUB -N
    #BSUB -o sendmeemail/error_%J.out 
    #BSUB -e sendmeemail/output_%J.err 
    
    module swap cuda/12.0
    module swap cudnn/v8.9.1.23-prod-cuda-12.X
    module swap python3/3.10.12
    
    python3 runtime.py \\
        --manifold {manifold} \\
        --lam {lam} \\
        --dim {d} \\
        --T {T} \\
        --method {method} \\
        --jax_lr_rate 0.01 \\
        --tol {tol} \\
        --max_iter 1000 \\
        --line_search_iter 100 \\
        --number_repeats 5 \\
        --timing_repeats 5 \\
        --seed 2712 \\
        --save_path ../runtimes/timing_gpu/ \\
        --svhn_path /work3/fmry/Data/SVHN/ \\
        --celeba_path /work3/fmry/Data/CelebA/
    ''')
    
    return

#%% Loop jobs

def loop_jobs(wait_time = 1.0):

    Ts = [100]
    lams = [1.0]
    methods = ["sgd", "rmsprop_momentum", "rmsprop", "adamax", "adam", "adagrad"]
    methods = methods + ["ProbGEORCE_LS", "ProbGEORCE_Adaptive"]
    #sphere
    runs = {"Sphere": [[2,3,5,10,20,50,100, 250, 500, 1000],1e-4],
            "Ellipsoid": [[2,3,5,10,20,50,100, 250, 500, 1000],1e-4],
            "SPDN": [[2,3],1e-4],
            "T2": [[2],1e-4],
            "Gaussian": [[2],1e-4],
            "Frechet": [[2],1e-4],
            "Cauchy": [[2],1e-4],
            "Pareto": [[2],1e-4],
            }
    
    #loop_over_jobs(lams, Ts, runs, methods, wait_time)

    Ts = [100]
    lams = [5.0, 10.0, 20.0, 50.0, 100.0]
    methods = ["sgd", "rmsprop_momentum", "rmsprop", "adamax", "adam", "adagrad"]
    methods = methods + ["ProbGEORCE_LS", "ProbGEORCE_Adaptive"]
    #sphere
    runs = {"Sphere": [[100],1e-4],
            }
    
    loop_over_jobs(lams, Ts, runs, methods, wait_time)
                                
#%% Submit call

def loop_over_jobs(lams, Ts, runs, methods, wait_time):
    
    for lam in lams:
        for T in Ts:
            for man, vals in runs.items():
                dims, tol = vals[0], vals[1]
                for d in dims:
                    for m in methods:
                        time.sleep(wait_time+np.abs(np.random.normal(0.0,1.,1)[0]))
                        generate_job(man, d, T, m, tol, lam)
                        try:
                            submit_job()
                        except:
                            time.sleep(100.0+np.abs(np.random.normal(0.0,1.,1)))
                            try:
                                submit_job()
                            except:
                                print(f"Job script with {lam}, {T}, {man}, {m}, {d}, {tol} failed!")
                                
    return

#%% main

if __name__ == '__main__':
    
    loop_jobs(1.0)