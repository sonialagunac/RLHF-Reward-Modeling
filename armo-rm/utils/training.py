from utils.utils import load_embeddings, eval_reward_bench, compute_stats
import pandas as pd
import numpy as np
import wandb, datasets, torch
from tqdm import tqdm
from torch.distributions import Beta

# --------------------------------------
# Training and Eval Regression Functions
# --------------------------------------


def train_regression(score_projection, beta_head, optimizer, dataloader, device, epoch, cfg_model):
    score_projection.train()
    beta_head.train()
    total_loss = 0

    for pos, neg, _, _, y in dataloader:
        pos, neg, y = pos.to(device), neg.to(device), y.to(device)
        optimizer.zero_grad()

        # Forward pass
        pos_out = score_projection(pos)
        neg_out = score_projection(neg)
        alpha, beta = beta_head(pos_out, neg_out)

        # Beta NLL Loss
        dist = Beta(alpha, beta)
        y = y.clamp(cfg_model.eps, 1 - cfg_model.eps) #Clamp y to avoid log(0) or log(1)
        nll = -dist.log_prob(y).mean()

        # Backprop
        nll.backward()
        optimizer.step()
        total_loss += nll.item()

    avg_loss = total_loss / len(dataloader)
    tqdm.write(f"Epoch {epoch}/{cfg_model.epochs_regression} - Regression NLL Loss: {avg_loss:.4f}")
    wandb.log({"regression_train_nll_loss": avg_loss, "epoch": epoch})
    return avg_loss

def validate_regression(score_projection, beta_head, dataloader, device):
    score_projection.eval()
    beta_head.eval()
    total_loss = 0

    with torch.no_grad():
        for pos, neg, _, _, y in dataloader:
            pos, neg, y = pos.to(device), neg.to(device), y.to(device)
            pos_out = score_projection(pos)
            neg_out = score_projection(neg)
            alpha, beta = beta_head(pos_out, neg_out)

            dist = Beta(alpha, beta)
            nll = -dist.log_prob(y).mean()
            total_loss += nll.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Regression Validation NLL Loss: {avg_loss:.4f}")
    wandb.log({"regression_val_nll_loss": avg_loss})
    return avg_loss

# ----------------------------------
# Training and Eval Gating Functions
# ----------------------------------
def train_gating(gating_network, score_projection, beta_head_gate, optimizer_gate, scheduler_gate, dataloader, device, epoch, cfg_model):
    gating_network.train()
    beta_head_gate.train()
    score_projection.eval()  # keep regression model fixed during gating training
    total_loss = 0
    for pos, neg, pos_prompt, neg_prompt, _ in dataloader:
        pos, neg, pos_prompt, neg_prompt = pos.to(device), neg.to(device), pos_prompt.to(device), neg_prompt.to(device)
        optimizer_gate.zero_grad()
        
        # Forward pass for positive samples
        weights_pos = gating_network(pos_prompt)
        concept_scores_pos = score_projection(pos)
        final_scores_pos = torch.sum(concept_scores_pos * weights_pos, dim=-1).unsqueeze(-1)
        
        # Forward pass for negative samples
        weights_neg = gating_network(neg_prompt)
        concept_scores_neg = score_projection(neg)
        final_scores_neg = torch.sum(concept_scores_neg * weights_neg, dim=-1).unsqueeze(-1)
        
        alpha, beta = beta_head_gate(final_scores_pos, final_scores_neg)

        # Beta NLL Loss
        dist = Beta(alpha, beta)
        nll = -dist.log_prob((1-cfg_model.eps)*torch.ones_like(final_scores_pos)).mean()

        # Backprop
        nll.backward()
        optimizer_gate.step()
        scheduler_gate.step()
        total_loss += nll.item()
    avg_loss = total_loss / len(dataloader)
    tqdm.write(f"Gating Training Step {epoch} - Loss: {avg_loss:.4f}")
    wandb.log({"gating_train_loss": avg_loss, "gating_epoch": epoch})
    return avg_loss


def validate_gating(gating_network, score_projection, beta_head_gate, val_dl, device, cfg_model):
    gating_network.eval()
    beta_head_gate.eval()
    score_projection.eval()
    with torch.no_grad():
        total_loss = 0
        for pos, neg, pos_prompt, neg_prompt, _ in val_dl:
            pos, neg, pos_prompt, neg_prompt = pos.to(device), neg.to(device), pos_prompt.to(device), neg_prompt.to(device)
            weights_val_pos = gating_network(pos_prompt)
            weights_val_neg = gating_network(neg_prompt)
            concept_scores_pos = score_projection(pos) 
            concept_scores_neg = score_projection(neg)
            final_scores_pos = torch.sum(concept_scores_pos * weights_val_pos, dim=-1).unsqueeze(-1) 
            final_scores_neg = torch.sum(concept_scores_neg * weights_val_neg, dim=-1).unsqueeze(-1)
            alpha, beta = beta_head_gate(final_scores_pos, final_scores_neg)

            # Beta NLL Loss
            dist = Beta(alpha, beta)
            nll = -dist.log_prob((1-cfg_model.eps)*torch.ones_like(final_scores_pos)).mean()
            total_loss += nll.item()
    avg_loss = total_loss / len(val_dl)
    print(f"Gating Validation NLL Loss: {avg_loss:.4f}")
    wandb.log({"gating_val_nll_loss": avg_loss})
    return avg_loss



def inference_active_learning(gating_network, score_projection, beta_head, beta_head_pref, test_dl, device):
    all_means_c, all_means_p,  all_uncertainties_c, all_uncertainties_p = [], [], [], []

    for batch_idx, (pos, neg, pos_prompt, neg_prompt, _) in enumerate(test_dl):
        pos, neg = pos.to(device), neg.to(device)
        pos_prompt, neg_prompt = pos_prompt.to(device), neg_prompt.to(device)

        with torch.no_grad():
            # Concept-level predictions
            pos_out = score_projection(pos)  # shape: [B, C]
            neg_out = score_projection(neg)  # shape: [B, C]
            alpha_c, beta_c = beta_head(pos_out, neg_out)  # shape: [B, C]
            mean_c, var_c = compute_stats(alpha_c, beta_c)  # shape: [B, C]

            # Preference-level predictions
            weights_pos = gating_network(pos_prompt)  # shape: [B, C]
            weights_neg = gating_network(neg_prompt)  # shape: [B, C]

            final_scores_pos = torch.sum(pos_out * weights_pos, dim=-1, keepdim=True)  # shape: [B, 1]
            final_scores_neg = torch.sum(neg_out * weights_neg, dim=-1, keepdim=True)  # shape: [B, 1]

            alpha_p, beta_p = beta_head_pref(final_scores_pos, final_scores_neg)  # shape: [B, 1]
            mean_p, var_p = compute_stats(alpha_p, beta_p)  # shape: [B, 1]

        # Store
        all_means_c.append(mean_c.cpu().numpy())  # [B, C]
        all_means_p.append(mean_p.cpu().numpy())  # [B, 1]
        all_uncertainties_c.append(var_c.cpu().numpy())  # [B, C]
        all_uncertainties_p.append(var_p.cpu().numpy())  # [B, 1]
        

    # Stack across batches
    means_c = np.vstack(all_means_c)  # shape: [N, C]
    means_p = np.vstack(all_means_p)  # shape: [N, 1]
    uncertainties_c = np.vstack(all_uncertainties_c)  # shape: [N, C]
    uncertainties_p = np.vstack(all_uncertainties_p)  # shape: [N, 1]
    
    return means_c, means_p, uncertainties_c, uncertainties_p


# ---------------------------
# RewardBench Evaluation
# ---------------------------
def reward_bench_eval(cfg, device, gating_network, score_projection, beta_head_pref):
    gating_network.eval()
    score_projection.eval()
    beta_head_pref.eval()
    print("Evaluating on RewardBench...")
    reward_bench_embeddings, reward_bench_prompt_embeddings = load_embeddings(cfg.reward_bench_embedding_path, device=device)

    with torch.no_grad():
        gating_weights_rb_0 = gating_network(reward_bench_prompt_embeddings[:, 0, :].squeeze())
        concept_scores_rb_0 = score_projection(reward_bench_embeddings[:, 0, :].squeeze()) 
        final_scores_rb_0 = torch.sum(concept_scores_rb_0 * gating_weights_rb_0, dim=-1).unsqueeze(-1)

        gating_weights_rb_1 = gating_network(reward_bench_prompt_embeddings[:, 1, :].squeeze())
        concept_scores_rb_1 = score_projection(reward_bench_embeddings[:, 1, :].squeeze()) 
        final_scores_rb_1 = torch.sum(concept_scores_rb_1 * gating_weights_rb_1, dim=-1).unsqueeze(-1)
        
        alpha, beta = beta_head_pref(final_scores_rb_0, final_scores_rb_1)
        dist = Beta(alpha, beta)
        mean = dist.mean # Using the mean as prediction, another alternative is to sample
        correct = (mean > 0.5).float().squeeze()

    # Load RewardBench dataset metadata
    reward_bench_ds = datasets.load_from_disk(cfg.path_reward_bench_data_filter)

    df_examples = pd.DataFrame({
        "subset": reward_bench_ds["subset"],
        "correct": correct.cpu().numpy()
    })
    scores_per_section, per_subset_metrics = eval_reward_bench(df_examples)
    print("RewardBench Scores:")
    print(pd.DataFrame([scores_per_section]))
    wandb.log({"rewardbench_scores": scores_per_section})