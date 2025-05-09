#!/bin/bash

#SBATCH --job-name=RLHF_interp
#SBATCH --output=../out_log/%j.out
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=0-20:00:00
#SBATCH -p gpu 
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-biomed-[01,03,07-10,12-16,22]

source ~/.bashrc
conda deactivate
conda activate swiss_ai
cd /cluster/home/slaguna/RLHF-Reward-Modeling/armo-rm/prepare_embeddings

python data_generation_transformers_batches.py --batch_index $1 --num_batches $2
