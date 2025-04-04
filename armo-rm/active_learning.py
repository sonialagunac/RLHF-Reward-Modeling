import torch
import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from safetensors.torch import load_file
import hydra
from omegaconf import DictConfig, OmegaConf

from models.networks import ScoreProjection, GatingNetwork, BetaHead
from utils.config import apply_path_defaults, init_wandb, set_seed
from utils.training import (
    train_regression, validate_regression,
    train_gating, validate_gating,
    inference_active_learning, reward_bench_eval,
)
from utils.data import get_dataloaders


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    set_seed(cfg.seed)
    apply_path_defaults(cfg)

    # Set device
    device = f"cuda:{cfg.device}" if cfg.device >= 0 and torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    init_wandb(cfg)

    # Load dataloaders
    train_dl, val_dl, input_dim, output_dim, concepts = get_dataloaders(cfg)
    test_dl = val_dl
    old_train_ds = train_dl.dataset

    score_projection = ScoreProjection(input_dim, output_dim).to(device)
    gating_network = GatingNetwork(
        in_features=input_dim,
        out_features=output_dim,
        n_hidden=cfg.model.n_hidden,
        hidden_dim=cfg.model.hidden_dim,
        dropout=cfg.model.dropout,
        temperature=cfg.model.temperature,
        logit_scale=cfg.model.logit_scale,
    ).to(device)
    beta_head = BetaHead(output_dim).to(device)
    beta_head_pref = BetaHead(1).to(device)

    # Params regression model
    reg_params = list(score_projection.parameters()) + list(beta_head.parameters())
    optimizer = torch.optim.AdamW(reg_params, lr=cfg.model.lr, weight_decay=cfg.model.weight_decay)

    gate_params = list(gating_network.parameters()) + list(beta_head_pref.parameters())
    optimizer_gate = torch.optim.AdamW(gate_params, lr=cfg.model.lr, weight_decay=cfg.model.weight_decay)
    scheduler_gate = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_gate, T_max=cfg.model.epochs_gating)

    if not cfg.experiment_name_al:
        experiment_name = cfg.experiment_name
        experiment_folder = os.path.join(cfg.data.output_dir, "models", f"{experiment_name}")
        os.makedirs(experiment_folder, exist_ok=True)
        print("Training regression model ...")
        for epoch in tqdm(range(cfg.model.epochs_regression)):
            train_regression(score_projection, beta_head, optimizer, updated_train_dl, device, epoch, cfg.model.epochs_regression)
        validate_regression(score_projection, beta_head, test_dl, device)

        print("Retraining gating network ...")
        for epoch in tqdm(range(cfg.model.epochs_gating)):
            train_gating(gating_network, score_projection, beta_head_pref, optimizer_gate, scheduler_gate, updated_train_dl, device, epoch)
        validate_gating(gating_network, score_projection, beta_head_pref, test_dl, device)

    else:
        # Load weights from pretrained reward model
        experiment_folder = os.path.join(
            cfg.data.output_dir, "models", cfg.experiment_name_al
        )
        model_ckpt_paths = {
            "score_projection": os.path.join(experiment_folder, f"regression_weights_{cfg.data.model_name}_{cfg.data.dataset_name}_labels_{cfg.data.labels_type}.pt"),
            "gating_network": os.path.join(experiment_folder, f"gating_network_{cfg.data.model_name}_{cfg.data.dataset_name}_labels_{cfg.data.labels_type}.pt"),
            "beta_head": os.path.join(experiment_folder, f"regression_weights_beta_{cfg.data.model_name}_{cfg.data.dataset_name}_labels_{cfg.data.labels_type}.pt"),
            "beta_head_pref": os.path.join(experiment_folder, f"gating_network_beta_{cfg.data.model_name}_{cfg.data.dataset_name}_labels_{cfg.data.labels_type}.pt"),
        }
        score_projection.load_state_dict(torch.load(model_ckpt_paths["score_projection"]))
        gating_network.load_state_dict(torch.load(model_ckpt_paths["gating_network"]))
        beta_head.load_state_dict(torch.load(model_ckpt_paths["beta_head"]))
        beta_head_pref.load_state_dict(torch.load(model_ckpt_paths["beta_head_pref"]))

    score_projection.eval()
    gating_network.eval()
    beta_head.eval()
    beta_head_pref.eval()

    # Run active learning
    uncertainties_p, uncertainties_c, all_indices = inference_active_learning(
        gating_network, score_projection, beta_head, beta_head_pref, test_dl, device
    )

    n_samples = cfg.get("n_samples", 10)
    selected_indices = np.random.choice(all_indices, size=n_samples, replace=False)

    # Extract new samples
    new_samples = [test_dl.dataset[idx] for idx in selected_indices]

    new_ds = TensorDataset(
        torch.stack([s[0] for s in new_samples]),
        torch.stack([s[1] for s in new_samples]),
        torch.stack([s[2] for s in new_samples]),
        torch.stack([s[3] for s in new_samples]),
        torch.stack([s[4] for s in new_samples]),
    )

    # Combine with old train data
    # TODO explore only using a batch of this and not the full training set
    updated_train_ds = ConcatDataset([old_train_ds, new_ds])
    updated_train_dl = DataLoader(updated_train_ds, batch_size=cfg.model.batch_size, shuffle=True)


    print("Rraining regression model on updated dataset...")
    for epoch in tqdm(range(cfg.model.epochs_regression)):
        train_regression(score_projection, beta_head, optimizer, updated_train_dl, device, epoch, cfg.model.epochs_regression)
    validate_regression(score_projection, beta_head, test_dl, device)

    print("Retraining gating network...")
    for epoch in tqdm(range(cfg.model.epochs_gating)):
        train_gating(gating_network, score_projection, beta_head_pref, optimizer_gate, scheduler_gate, updated_train_dl, device, epoch)
    validate_gating(gating_network, score_projection, beta_head_pref, test_dl, device)

    # Optionally run RewardBench evaluation
    if cfg.eval_reward_bench:
        reward_bench_eval(cfg, device, gating_network, score_projection, beta_head)


if __name__ == "__main__":
    main()