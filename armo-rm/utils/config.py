# utils/config.py

import os
import torch
import random
import numpy as np
import wandb

# ---------------------------
# Set up Weights & Biases
# ---------------------------
def init_wandb(cfg):
    wandb.init(
        project="interpretable_rewards",
        entity=cfg.wandb_entity,
        config=dict(cfg),
        dir=cfg.wandb_path
    )

# ---------------------------
# Set up Random Seeds
# ---------------------------
def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if using multi-GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Seeds set to {seed}")

# ---------------------------
# Set up Paths
# ---------------------------
def apply_path_defaults(cfg):
    """
    Fill in default paths if they are null in cfg.data
    """
    data = cfg.data

    if data.embeddings_dir is None:
        data.embeddings_dir = os.path.join(
            data.output_dir,
            "embeddings",
            data.model_name,
            f"{data.dataset_name}-train.safetensors"
        )

    if data.labels_dir is None:
        data.labels_dir = os.path.join(
            data.output_dir,
            "labels",
            data.labels_type,
            f"{data.dataset_name}_combined.safetensors"
        )

    if cfg.reward_bench_embedding_path is None:
        cfg.reward_bench_embedding_path = os.path.join(
            data.output_dir,
            "embeddings",
            data.model_name,
            "reward_bench-filtered.safetensors"
        )
