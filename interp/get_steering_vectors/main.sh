#!/bin/bash -l

# This script is set up so that you can either qsub it or run it interactively

#$ -P vkolagrp
#$ -l h_rt=6:00:00
#$ -pe omp 8
#$ -l mem_per_core=2G
#$ -l gpus=1

# GPU capability, must be at least 8 for this project
#$ -l gpu_c=8
#$ -m bea

# export TORCH_HOME=/projectnb/vkolagrp/skowshik/.cache/torch
# export VLLM_CACHE_DIR=/projectnb/vkolagrp/skowshik/.cache/vllm
export HF_HOME=/projectnb/vkolagrp/skowshik/.cache/
export TORCH_HOME=/projectnb/cs599m1/students/skowshik/.cache/torch
# export VLLM_CACHE_DIR=/projectnb/cs599m1/students/skowshik/.cache/vllm
# export VLLM_SKIP_P2P_CHECK=1

# conda activate /projectnb/cs599m1/students/skowshik/.cache/conda_envs/cs599m1_env

python main.py --config config.yml

# qrsh -P vkolagrp -l gpus=1 -l gpu_c=8 -l h_rt=10:00:00