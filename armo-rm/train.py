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
from torch.utils.data import DataLoader, TensorDataset

from models.networks import GatingNetwork
from utils.training import train_regression, validate_regression, train_gating, validate_gating, reward_bench_eval
from utils.config import parse_args, init_wandb, set_offline_paths, set_seed


def main():
    args = parse_args()

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
    # Load embeddings and labels from safetensors
    # ---------------------------
    print("Loading embeddings...")
    embed_data = load_file(args.embeddings_dir)
    embeddings = embed_data["embeddings"].float()
    prompt_embeddings = embed_data["prompt_embeddings"].float()
    
    label_data = load_file(args.labels_dir)
    concept_labels = label_data["concepts_label"].float().transpose(0,1)
    
    if args.labels_type == 'hugging_face':
        concepts = ["helpfulness", "honesty", "instruction_following", "truthfulness"]
    else:
        concepts = ["helpfulness", "correctness", "coherence", "complexity", "verbosity",
                    "overall_score", "instruction_following", "truthfulness", "honesty",  
                    "is_safe", "score", "overall_quality", "judge_lm", "style", "explanation", "readability"]
    
    # Get pairwise embeddings
    pos_embeddings = embeddings[:, 0]       
    neg_embeddings = embeddings[:, 1] 
    pos_prompt_embeddings = prompt_embeddings[:, 0]   
    neg_prompt_embeddings = prompt_embeddings[:, 1]
    
    print("Embeddings shape:", embeddings.shape)
    print("Concept labels shape:", concept_labels.shape)
    
    # ---------------------------
    # Train/Val split & DataLoaders
    # ---------------------------
    X_pos_train, X_pos_val, X_neg_train, X_neg_val, X_pos_prompt_train, X_pos_prompt_val, X_neg_prompt_train, X_neg_prompt_val, y_train, y_val = train_test_split(
        pos_embeddings, neg_embeddings, pos_prompt_embeddings, neg_prompt_embeddings, concept_labels, test_size=0.2, random_state=42
    )
    
    train_ds = TensorDataset(X_pos_train, X_neg_train, X_pos_prompt_train, X_neg_prompt_train, y_train)
    val_ds = TensorDataset(X_pos_val, X_neg_val, X_pos_prompt_val, X_neg_prompt_val, y_val)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size)
    
    # ---------------------------
    # Initialize Regression Model and Optimizer
    # ---------------------------
    input_dim = pos_embeddings.shape[1]
    output_dim = concept_labels.shape[1]  
    regression_model = nn.Linear(input_dim, output_dim, bias=False).to(device)
    optimizer = torch.optim.AdamW(regression_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()
    
    # ---------------------------
    # Initialize Gating Network and Optimizer
    # ---------------------------
    gating_network = GatingNetwork(
        in_features=X_pos_prompt_train.shape[-1],
        out_features=y_train.shape[-1],
        n_hidden=args.n_hidden,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        temperature=args.temperature,
        logit_scale=args.logit_scale,
    ).to(device)
    
    optimizer_gate = torch.optim.AdamW(gating_network.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler_gate = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_gate, T_max=args.epochs_gating)
    loss_gate_fn = nn.BCEWithLogitsLoss()
    
    # ---------------------------
    # Regression Training Loop
    # ---------------------------
    print("Training multivariate regression model on concept scores...")
    for epoch in tqdm(range(args.epochs_regression)):
        train_regression(regression_model, optimizer, loss_fn, train_dl, device, epoch, args.epochs_regression)
    validate_regression(regression_model, loss_fn, val_dl, device)
    
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
        train_gating(gating_network, regression_model, optimizer_gate, loss_gate_fn, scheduler_gate, train_dl, device, epoch)
    validate_gating(gating_network, regression_model, val_dl, device)
    
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