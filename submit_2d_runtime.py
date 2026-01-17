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
    
    os.system("bsub < submit_2d_runtime.sh")
    
    return

#%% Generate jobs

def generate_job(runtime_type, model_type, computation, method, geodesic_method, lam, tol=1e-4):

    with open ('submit_2d_runtime.sh', 'w') as rsh:
        rsh.write(f'''\
    #! /bin/bash
    #BSUB -q gpuv100
    #BSUB -J {model_type}_{computation}_{method}_{geodesic_method}_{lam}
    #BSUB -n 4
    #BSUB -gpu "num=1:mode=exclusive_process"
    #BSUB -W 24:00
    #BSUB -R "span[hosts=1]"
    #BSUB -R "rusage[mem=32GB]"
    #BSUB -u fmry@dtu.dk
    #BSUB -B
    #BSUB -N
    #BSUB -o ../output_folder/output_%J.out 
    #BSUB -e ../error_folder/error_%J.err 
    
    module swap python3/3.10.12
    module swap cuda/12.0
    module swap cudnn/v8.9.1.23-prod-cuda-12.X
    
    python3 run_gen2d_tests.py \\
        --runtime_type {runtime_type} \\
        --model_type {model_type} \\
        --computation {computation} \\
        --method {method} \\
        --geodesic_method {geodesic_method} \\
        --lam {lam} \\
        --lam_identity 1.0 \\
        --grid_size 100 \\
        --N_grid 100 \\
        --max_iter 1000 \\
        --tol {tol} \\
        --number_repeats 5 \\
        --timing_repeats 5 \\
        --seed 2712 \\
        --device cuda \\
        --model_path ../models/gen2d/ \\
        --save_path ../gen2d_results/gpu/ \\
    ''')
    
    return

#%% Loop jobs

def loop_jobs(wait_time = 1.0):
    
    model_type = ['ebm', 'nf', 'ar', 'vae']
    computation = ['ivp', 'bvp', 'mean']
    method = ['ProbGEORCE', 'Linear', 'SLERP', 'Fisher-Rao', 'Fisher-Rao-Reg', 'Jacobian-Metric', 'Jacobian-Metric-Reg', 'Inverse-Density', 'Generative-Metric', 'Monge-Metric']
    geodesic_method = ['ProbGEORCE_Adaptive', 'ProbGEORCE_LS', 'Adam', 'SGD', 'RMSprop', 'AdamW', 'LBFGS']
    
    #method = ['Generative-Metric']
    #model_type = ['vae']
    #computation = ['ivp', 'bvp']
    run_model("metrics", model_type, computation, method, ['ProbGEORCE_Adaptive'], [20.0], wait_time)
    #run_model("grid", ['ebm'], ['bvp'], ['ProbGEORCE'], ['ProbGEORCE_Adaptive'], [0.0, 5.0, 20.0, 100.0], wait_time)
    #run_model("runtime", model_type, ['bvp'], ['ProbGEORCE'], geodesic_method, [0.1, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0], wait_time)
    
    return
                            
def run_model(runtime_type, model_type, computation, method, geodesic_method, lams, wait_time):
    
    for lam in lams:
        for mmodel_type in model_type:
            for mcomputation in computation:
                for mmethod in method:
                    for mgeodesic_method in geodesic_method:
                        time.sleep(wait_time+np.abs(np.random.normal(0.0,1.,1)[0]))
                        generate_job(runtime_type = runtime_type,
                                     model_type = mmodel_type, 
                                     computation = mcomputation, 
                                     method = mmethod, 
                                     geodesic_method = mgeodesic_method,
                                     lam = lam,
                                     tol=1e-4,
                                     )
                        try:
                            submit_job()
                        except:
                            time.sleep(100.0+np.abs(np.random.normal(0.0,1.,1)))
                            try:
                                submit_job()
                            except:
                                print(f"Job script failed!")


#%% main

if __name__ == '__main__':
    
    loop_jobs(1.0)
