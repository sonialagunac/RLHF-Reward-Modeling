import torch
import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import hydra
from omegaconf import DictConfig, OmegaConf

from utils.config import apply_path_defaults, init_wandb, set_seed, get_models
from utils.training import (
    train_regression, validate_regression,
    train_gating, validate_gating,
    inference_active_learning, reward_bench_eval,
)
from utils.data import get_dataloaders


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Set reproducibility seed
    set_seed(cfg.seed)

    # Fill derived paths (embeddings, labels, etc.)
    paths_weights = apply_path_defaults(cfg)

    # Set device
    device = f"cuda:{cfg.device}" if cfg.device >= 0 and torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Init Weights & Biases logging
    init_wandb(cfg)

    # Load dataset
    train_dl, val_dl, test_dl, input_dim, output_dim, concepts = get_dataloaders(cfg)

    # Initialize models
    score_projection, beta_head, optimizer, gating_network, beta_head_pref, optimizer_gate, scheduler_gate = get_models(
        cfg, input_dim, output_dim, device
    )

    if not cfg.experiment_name_al:
        # Training reward model for the first time
        if cfg.store_weights:
            experiment_name = cfg.experiment_name
            experiment_folder = os.path.join(cfg.data.output_dir, "models", f"{experiment_name}")
            os.makedirs(experiment_folder, exist_ok=True)
        
        print("Training regression model ...")
        for epoch in tqdm(range(cfg.model.epochs_regression)):
            train_regression(score_projection, beta_head, optimizer, train_dl, device, epoch, cfg.model)
        validate_regression(score_projection, beta_head, val_dl, device)
        # Save regression model
        if cfg.store_weights:
            torch.save(score_projection.state_dict(), os.path.join(experiment_folder, paths_weights[0]))
            torch.save(beta_head.state_dict(), os.path.join(experiment_folder, paths_weights[1]))
        
        print("Training gating network ...")
        for epoch in tqdm(range(cfg.model.epochs_gating)):
            train_gating(gating_network, score_projection, beta_head_pref, optimizer_gate, scheduler_gate, train_dl, device, epoch, cfg.model)
        validate_gating(gating_network, score_projection, beta_head_pref, val_dl, device, cfg.model)
        # Save gating model
        if cfg.store_weights:
            torch.save(gating_network.state_dict(), os.path.join(experiment_folder, paths_weights[2]))
            torch.save(beta_head_pref.state_dict(), os.path.join(experiment_folder, paths_weights[3]))
    
    else:
        # Load weights from pretrained reward model
        experiment_folder = os.path.join(
            cfg.data.output_dir, "models", cfg.experiment_name_al
        )
        model_ckpt_paths = {
            "score_projection": os.path.join(experiment_folder, paths_weights[0]),
            "gating_network": os.path.join(experiment_folder, paths_weights[2]),
            "beta_head": os.path.join(experiment_folder, paths_weights[1]),
            "beta_head_pref": os.path.join(experiment_folder, paths_weights[]),
        }
        score_projection.load_state_dict(torch.load(model_ckpt_paths["score_projection"]))
        gating_network.load_state_dict(torch.load(model_ckpt_paths["gating_network"]))
        beta_head.load_state_dict(torch.load(model_ckpt_paths["beta_head"]))
        beta_head_pref.load_state_dict(torch.load(model_ckpt_paths["beta_head_pref"]))

    score_projection.eval()
    gating_network.eval()
    beta_head.eval()
    beta_head_pref.eval()

    # -----------------------------
    # ACTIVE LEARNING LOOP
    # -----------------------------
    current_train_ds = train_dl.dataset
    best_val_loss = validate_gating(gating_network, score_projection, beta_head_pref, val_dl, device, cfg.model)

    patience_counter = 0

    for iteration in range(cfg.model.max_iters):
        print(f"\n===== Active Learning Iteration {iteration+1} =====")

        uncertainties_p, uncertainties_c, all_indices = inference_active_learning(
            gating_network, score_projection, beta_head, beta_head_pref, test_dl, device
        )

        # TODO do function so that you can select from the hydra configs what acq function to use
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

        # Combine with prior train data
        # TODO explore only using a batch of this and not the full training set
        current_train_ds = ConcatDataset([current_train_ds, new_ds])
        updated_train_dl = DataLoader(current_train_ds, batch_size=cfg.model.batch_size, shuffle=True)

        print("Retraining regression model on updated dataset...")
        for epoch in tqdm(range(cfg.model.epochs_regression)):
            train_regression(score_projection, beta_head, optimizer, updated_train_dl, device, epoch, cfg.model)
        validate_regression(score_projection, beta_head, val_dl, device)

        # Save regression model
        if cfg.store_weights:
            torch.save(score_projection.state_dict(), os.path.join(experiment_folder, f"post_AL_{paths_weights[0]}"))
            torch.save(beta_head.state_dict(), os.path.join(experiment_folder, f"post_AL_{paths_weights[1]}"))

        print("Retraining gating network...")
        for epoch in tqdm(range(cfg.model.epochs_gating)):
            train_gating(gating_network, score_projection, beta_head_pref, optimizer_gate, scheduler_gate, updated_train_dl, device, epoch, cfg.model)
        new_val_loss = validate_gating(gating_network, score_projection, beta_head_pref, val_dl, device, cfg.model)

        # Save gating model
        if cfg.store_weights:
            torch.save(gating_network.state_dict(), os.path.join(experiment_folder, f"post_AL_{paths_weights[2]}"))
            torch.save(beta_head_pref.state_dict(), os.path.join(experiment_folder, f"post_AL_{paths_weights[3]}"))

        # Check for early stopping
        improvement = best_val_loss - new_val_loss
        print(f"Improvement: {improvement:.4f}") # Note the improvemnt for now is in the preference loss
        if iteration + 1 >= cfg.model.min_rounds:
            if improvement < cfg.model.min_improvement:
                patience_counter += 1
                print(f"Patience counter: {patience_counter}/{cfg.model.patience_limit}")
                if patience_counter >= cfg.model.patience_limit:
                    print("Stopping early due to lack of improvement.")
                    break
            else:
                patience_counter = 0  # reset if improvement is good
        best_val_loss = new_val_loss

    # Optionally run RewardBench evaluation
    if cfg.eval_reward_bench:
        reward_bench_eval(cfg, device, gating_network, score_projection, beta_head_pref)


if __name__ == "__main__":
    main()