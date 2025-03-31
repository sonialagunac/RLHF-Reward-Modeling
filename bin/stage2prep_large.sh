#!/bin/bash

#SBATCH --job-name=RLHF_interp
#SBATCH --output=../out_log/%j.out
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --time=1-10:00:00
#SBATCH -p gpu 
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-biomed-[01,03,07-10,12-16,22]

source ~/.bashrc
conda deactivate
conda activate swiss_ai

# --exclude=gpu-biomed-[03,07-10,12-13,22] #to only exc the 2080s
cd /cluster/home/slaguna/RLHF-Reward-Modeling/armo-rm/prepare_embeddings/
python stage-2_prepare_large.py "$@"
