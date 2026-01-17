    #! /bin/bash
    #BSUB -q gpuv100
    #BSUB -J vae_bvp_ProbGEORCE_LBFGS_100.0
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
    
    python3 run_gen2d_tests.py \
        --runtime_type runtime \
        --model_type vae \
        --computation bvp \
        --method ProbGEORCE \
        --geodesic_method LBFGS \
        --lam 100.0 \
        --lam_identity 1.0 \
        --grid_size 100 \
        --N_grid 100 \
        --max_iter 1000 \
        --tol 0.0001 \
        --number_repeats 5 \
        --timing_repeats 5 \
        --seed 2712 \
        --device cuda \
        --model_path ../models/gen2d/ \
        --save_path ../gen2d_results/gpu/ \
    