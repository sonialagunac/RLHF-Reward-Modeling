#!/bin/bash

#SBATCH --job-name=RLHF_interp
#SBATCH --output=../out_log/%j.out
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=2-00:00:00
#SBATCH -p gpu 
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-biomed-[01,03,07-10,12-16,22]

source ~/.bashrc
conda deactivate
conda activate swiss_ai

# --exclude=gpu-biomed-[03,07-10,12-13,22] #to only exc the 2080s
cd /cluster/home/slaguna/RLHF-Reward-Modeling/armo-rm/
python stage-1_prepare.py --n_shards 1 --shard_idx 1 --device 0 "$@"
