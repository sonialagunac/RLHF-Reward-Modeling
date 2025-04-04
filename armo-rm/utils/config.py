# utils/config.py

import os
import torch
import random
import numpy as np
import wandb
from models.networks import ScoreProjection, GatingNetwork, BetaHead

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
    Fill in default paths in cfg.data if they are not already set.
    Also returns standard filename prefixes for model weights.
    """
    data = cfg.data

    # Fill default embedding and label paths
    data.embeddings_dir = data.embeddings_dir or os.path.join(
        data.output_dir, "embeddings", data.model_name, f"{data.dataset_name}-train.safetensors"
    )

    data.labels_dir = data.labels_dir or os.path.join(
        data.output_dir, "labels", data.labels_type, f"{data.dataset_name}_combined.safetensors"
    )

    cfg.reward_bench_embedding_path = cfg.reward_bench_embedding_path or os.path.join(
        data.output_dir, "embeddings", data.model_name, "reward_bench-filtered.safetensors"
    )

    # Compose common model weight filenames
    base = f"{data.model_name}_{data.dataset_name}_labels_{data.labels_type}.pt"
    paths_weights = [
        f"regression_weights_{base}",
        f"regression_weights_beta_{base}",
        f"gating_network_{base}",
        f"gating_network_beta_{base}",
    ]

    return paths_weights

    
# ---------------------------
# Initialize Models
# ---------------------------
def get_models(cfg, input_dim, output_dim, device):

    # Initialize regression model + optimizer
    score_projection = ScoreProjection(input_dim, output_dim).to(device)
    beta_head = BetaHead(output_dim).to(device)
    reg_params = list(score_projection.parameters()) + list(beta_head.parameters())
    optimizer = torch.optim.AdamW(reg_params, lr=cfg.model.lr, weight_decay=cfg.model.weight_decay)

    # Initialize gating network + optimizer
    gating_network = GatingNetwork(
        in_features=input_dim,
        out_features=output_dim,
        n_hidden=cfg.model.n_hidden,
        hidden_dim=cfg.model.hidden_dim,
        dropout=cfg.model.dropout,
        temperature=cfg.model.temperature,
        logit_scale=cfg.model.logit_scale,
    ).to(device)

    beta_head_pref = BetaHead(1).to(device)
    gate_params = list(gating_network.parameters()) + list(beta_head_pref.parameters())
    optimizer_gate = torch.optim.AdamW(gate_params, lr=cfg.model.lr, weight_decay=cfg.model.weight_decay)
    scheduler_gate = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_gate, T_max=cfg.model.epochs_gating)

    return score_projection, beta_head, optimizer, gating_network, beta_head_pref, optimizer_gate, scheduler_gate
