import torch
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from safetensors.torch import load_file
import numpy as np
import random
import os
from tqdm import tqdm
from models.networks import ScoreProjection, GatingNetwork, BetaHead
from utils.config import parse_args, set_default_paths, set_offline_paths, set_seed, init_wandb
from utils.training import train_regression, validate_regression, train_gating, validate_gating, inference_active_learning, reward_bench_eval
from utils.data import get_dataloaders


def main():
    args = parse_args()
    args = set_default_paths(args)
    if args.offline:
        args = set_offline_paths(args)
    set_seed(args.seed)
    device = f"cuda:{args.device}" if args.device >= 0 and torch.cuda.is_available() else "cpu"
    init_wandb(args)

   # TODO instead of train_dl, split in test val train and use the test here!
    train_dl, val_dl, input_dim, output_dim, concepts = get_dataloaders(args) 
    test_dl = val_dl
    old_train_ds = train_dl.dataset   # Assume you have stored model weights paths
    #TODO automate this path
    experiment_name_dir = "default_experiment_20250403_142410"
    experiment_folder = os.path.join(args.output_dir, "models", experiment_name_dir)
    model_ckpt_paths = {
        "score_projection":  os.path.join(experiment_folder, f"regression_weights_{args.model_name}_{args.dataset_name}_labels_{args.labels_type}.pt"),
        "gating_network": os.path.join(experiment_folder, f"gating_network_{args.model_name}_{args.dataset_name}_labels_{args.labels_type}.pt"),
        "beta_head": os.path.join(experiment_folder, f"regression_weights_beta_{args.model_name}_{args.dataset_name}_labels_{args.labels_type}.pt"),
        "beta_head_pref": os.path.join(experiment_folder, f"gating_network_beta_{args.model_name}_{args.dataset_name}_labels_{args.labels_type}.pt"),
    }
    
    # Load models
    score_projection = ScoreProjection(input_dim, output_dim).to(device)
    gating_network = GatingNetwork(
        in_features=input_dim,
        out_features=output_dim,
        n_hidden=args.n_hidden,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        temperature=args.temperature,
        logit_scale=args.logit_scale,
    ).to(device)
    beta_head = BetaHead(output_dim).to(device)
    beta_head_pref = BetaHead(1).to(device)

    # Load weights
    score_projection.load_state_dict(torch.load(model_ckpt_paths['score_projection']))
    gating_network.load_state_dict(torch.load(model_ckpt_paths['gating_network']))
    beta_head.load_state_dict(torch.load(model_ckpt_paths['beta_head']))
    beta_head_pref.load_state_dict(torch.load(model_ckpt_paths['beta_head_pref']))

    score_projection.eval()
    gating_network.eval()
    beta_head.eval()
    beta_head_pref.eval()

    # Active learning loop
    uncertainties_p, uncertainties_c, all_indices = inference_active_learning(gating_network, score_projection, beta_head, beta_head_pref, test_dl, device)

    # For now: random selection of samples, define your own acquisition function
    n_samples = 10 # TODO make this an argument variable
    selected_indices = np.random.choice(all_indices, size=n_samples, replace=False)

    # Get new samples and ground truth labels
    new_samples = []
    for idx in selected_indices:
        pos, neg, pos_prompt, neg_prompt, y = test_dl.dataset[idx]
        new_samples.append((pos, neg, pos_prompt, neg_prompt, y))


    # Create new dataset
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
    updated_train_dl = DataLoader(updated_train_ds, batch_size=args.batch_size, shuffle=True)

    # Retrain model
    print("Retraining on updated dataset...")

    params = list(score_projection.parameters()) + list(beta_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    for epoch in tqdm(range(args.epochs_regression)):
        train_regression(score_projection, beta_head, optimizer, updated_train_dl, device, epoch, args.epochs_regression)
    validate_regression(score_projection, beta_head, test_dl, device)

    # Retrain Gating Network
    optimizer_gate = torch.optim.AdamW(gating_network.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler_gate = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_gate, T_max=args.epochs_gating)

    for epoch in tqdm(range(args.epochs_gating)):
        train_gating(gating_network, score_projection, beta_head_pref, optimizer_gate, scheduler_gate, updated_train_dl, device, epoch)
    validate_gating(gating_network, score_projection, beta_head_pref, test_dl, device)

    if args.eval_reward_bench:
        reward_bench_eval(args, device, gating_network, regression_model)



if __name__ == "__main__":
    main()