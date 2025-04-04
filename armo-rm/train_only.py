# ======================
# Interpretable Rewards 
# ======================

import os
import torch
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf

from models.networks import GatingNetwork, ScoreProjection, BetaHead
from utils.training import train_regression, validate_regression, train_gating, validate_gating, reward_bench_eval
from utils.config import init_wandb, set_seed, apply_path_defaults
from utils.data import get_dataloaders


@hydra.main(version_base=None,config_path="configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Set reproducibility seed
    set_seed(cfg.seed)

    # Fill derived paths (embeddings, labels, etc.)
    apply_path_defaults(cfg)

    # Set device
    device = f"cuda:{cfg.device}" if cfg.device >= 0 and torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Init Weights & Biases logging
    init_wandb(cfg)


    # Load dataset
    train_dl, val_dl, _, input_dim, output_dim, concepts = get_dataloaders(cfg)

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

    # Create output directory for experiment
    if cfg.store_weights:
        experiment_name = cfg.experiment_name
        experiment_folder = os.path.join(cfg.data.output_dir, "models", f"{experiment_name}")
        os.makedirs(experiment_folder, exist_ok=True)

    # ---------------------------
    # Train regression model
    # ---------------------------
    print("Training multivariate regression model on concept scores...")
    for epoch in tqdm(range(cfg.model.epochs_regression)):
        train_regression(score_projection, beta_head, optimizer, train_dl, device, epoch, cfg.model)
    validate_regression(score_projection, beta_head, val_dl, device)

    # Save regression model
    if cfg.store_weights:
        torch.save(score_projection.state_dict(), os.path.join(experiment_folder, f"regression_weights_{cfg.data.model_name}_{cfg.data.dataset_name}_labels_{cfg.data.labels_type}.pt"))
        torch.save(beta_head.state_dict(), os.path.join(experiment_folder, f"regression_weights_beta_{cfg.data.model_name}_{cfg.data.dataset_name}_labels_{cfg.data.labels_type}.pt"))

    # ---------------------------
    # Train gating network
    # ---------------------------
    print("Training gating network...")
    for epoch in tqdm(range(cfg.model.epochs_gating)):
        train_gating(gating_network, score_projection, beta_head_pref, optimizer_gate, scheduler_gate, train_dl, device, epoch, cfg.model)
    validate_gating(gating_network, score_projection, beta_head_pref, val_dl, device, cfg.model)

    # Save gating model
    if cfg.store_weights:
        torch.save(gating_network.state_dict(), os.path.join(experiment_folder, f"gating_network_{cfg.data.model_name}_{cfg.data.dataset_name}_labels_{cfg.data.labels_type}.pt"))
        torch.save(beta_head_pref.state_dict(), os.path.join(experiment_folder, f"gating_network_beta_{cfg.data.model_name}_{cfg.data.dataset_name}_labels_{cfg.data.labels_type}.pt"))

    # RewardBench evaluation
    if cfg.eval_reward_bench:
        reward_bench_eval(cfg, device, gating_network, score_projection, beta_head)


if __name__ == "__main__":
    main()