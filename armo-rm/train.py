# ===============================
# Interpretable Rewards: Regression + Gating Network
# ===============================

import os
import torch
import datetime
from tqdm import tqdm
from safetensors.torch import load_file
from sklearn.model_selection import train_test_split
from torch import nn

from models.networks import GatingNetwork, ScoreProjection, BetaHead
from utils.training import train_regression, validate_regression, train_gating, validate_gating, reward_bench_eval
from utils.config import parse_args, set_default_paths, init_wandb, set_offline_paths, set_seed
from utils.data import get_dataloaders

def main():
    args = parse_args()
    args = set_default_paths(args)

    # Set random seed for reproducibility
    set_seed(args.seed)

    # Set offline (cluster) paths if required
    if args.offline:
        args = set_offline_paths(args)
    
    if args.store_weights:
        # Create output directory for this experiment
        experiment_name = args.experiment_name if hasattr(args, "experiment_name") and args.experiment_name else "default_experiment"
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_folder = os.path.join(args.output_dir, "models", f"{experiment_name}_{timestamp}")
        os.makedirs(experiment_folder, exist_ok=True)

    # Initialize Weights & Biases
    init_wandb(args)
    
    # Set device
    device = f"cuda:{args.device}" if args.device >= 0 and torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # ---------------------------
    # Load embeddings and labels from safetensors, create dataloaders
    # ---------------------------
    train_dl, val_dl, input_dim, output_dim, concepts = get_dataloaders(args)
    
    # ---------------------------
    # Initialize Regression Model and Optimizer
    # ---------------------------
    score_projection = ScoreProjection(input_dim, output_dim).to(device)
    beta_head = BetaHead(output_dim).to(device)
    params = list(score_projection.parameters()) + list(beta_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    # regression_model = nn.Linear(input_dim, output_dim, bias=False).to(device)
    # optimizer = torch.optim.AdamW(regression_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # loss_fn = nn.MSELoss()
    
    # ---------------------------
    # Initialize Gating Network and Optimizer
    # ---------------------------
    gating_network = GatingNetwork(
        in_features=input_dim,
        out_features=output_dim,
        n_hidden=args.n_hidden,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        temperature=args.temperature,
        logit_scale=args.logit_scale,
    ).to(device)
    beta_head_gate = BetaHead(1).to(device)
    optimizer_gate = torch.optim.AdamW(gating_network.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler_gate = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_gate, T_max=args.epochs_gating)
    # loss_gate_fn = nn.BCEWithLogitsLoss()
    
    # ---------------------------
    # Regression Training Loop
    # ---------------------------
    print("Training multivariate regression model on concept scores...")
    for epoch in tqdm(range(args.epochs_regression)):
        train_regression(score_projection, beta_head, optimizer, train_dl, device, epoch, args.epochs_regression)
    validate_regression(score_projection, beta_head, val_dl, device)
    
    # ---------------------------
    # Save Regression Weights
    # ---------------------------
    if args.store_weights:
        regression_weights = regression_model.weight.detach().cpu()  
        regression_save_path = os.path.join(experiment_folder, f"regression_weights_{args.model_name}_{args.dataset_name}_labels_{args.labels_type}.pt")
        torch.save({"weight": regression_weights}, regression_save_path) 
        print(f"Saved regression weights to {regression_save_path}")
    
    # ---------------------------
    # Gating Network Training Loop
    # ---------------------------
    print("Training gating network...")
    for epoch in tqdm(range(args.epochs_gating)):
        # Here we optimize the gating network while keeping the regression model fixed.
        train_gating(gating_network, score_projection, beta_head_gate, optimizer_gate, scheduler_gate, train_dl, device, epoch)
    validate_gating(gating_network, score_projection, beta_head_gate, val_dl, device)
    
    # ---------------------------
    # Save Gating Network
    # ---------------------------
    if args.store_weights:
        gating_save_path = os.path.join(experiment_folder, f"gating_network_{args.model_name}_{args.dataset_name}_labels_{args.labels_type}.pt")
        torch.save(gating_network.state_dict(), gating_save_path)
        print(f"Gating Network saved to {gating_save_path}")
    
    if args.eval_reward_bench:
        reward_bench_eval(args, device, gating_network, regression_model)


if __name__ == "__main__":
    main()