from utils.utils import load_embeddings, eval_reward_bench, compute_variance
import pandas as pd
import numpy as np
import wandb, datasets, torch, tqdm
from torch.distributions import Beta

# --------------------------------------
# Training and Eval Regression Functions
# --------------------------------------


def train_regression(score_projection, beta_head, optimizer, dataloader, device, epoch, total_epochs):
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
        eps = 1e-4 #TODO make it a parameter
        y = y.clamp(eps, 1 - eps) #Clamp y to avoid log(0) or log(1)
        nll = -dist.log_prob(y).mean()

        # Backprop
        nll.backward()
        optimizer.step()
        total_loss += nll.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch}/{total_epochs} - Regression NLL Loss: {avg_loss:.4f}")
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
def train_gating(gating_network, score_projection, beta_head_gate, optimizer_gate, scheduler_gate, dataloader, device, epoch):
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
        eps = 1e-4 #TODO make it a parameter
        nll = -dist.log_prob((1-eps)*torch.ones_like(final_scores_pos)).mean()

        # Backprop
        nll.backward()
        optimizer_gate.step()
        scheduler_gate.step()
        total_loss += nll.item()
    avg_loss = total_loss / len(dataloader)
    print(f"Gating Training Step {epoch} - Loss: {avg_loss:.4f}")
    wandb.log({"gating_train_loss": avg_loss, "gating_epoch": epoch})
    return avg_loss


def validate_gating(gating_network, score_projection, beta_head_gate, val_dl, device):
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
            eps = 1e-4 #TODO make it a parameter
            nll = -dist.log_prob((1-eps)*torch.ones_like(final_scores_pos)).mean()
            total_loss += nll.item()
    avg_loss = total_loss / len(val_dl)
    print(f"Gating Validation NLL Loss: {avg_loss:.4f}")
    wandb.log({"gating_val_nll_loss": avg_loss})
    return avg_loss



def inference_active_learning(gating_network, score_projection, beta_head, beta_head_pref, test_dl, device):
    uncertainties_p = []
    uncertainties_c = []
    all_indices = []

    # Inference on test set
    for i, (pos, neg, pos_prompt, neg_prompt, y) in enumerate(test_dl):
        pos, neg, pos_prompt, neg_prompt = pos.to(device), neg.to(device), pos_prompt.to(device), neg_prompt.to(device)
        with torch.no_grad():
            pos_out = score_projection(pos)
            neg_out = score_projection(neg)
            alpha_c, beta_c = beta_head(pos_out, neg_out)
            var_c = compute_variance(alpha_c, beta_c)

            weights_pos = gating_network(pos_prompt)
            weights_neg = gating_network(neg_prompt)
            concept_scores_pos = pos_out
            concept_scores_neg = neg_out

            final_scores_pos = (concept_scores_pos * weights_pos).sum(dim=-1, keepdim=True)
            final_scores_neg = (concept_scores_neg * weights_neg).sum(dim=-1, keepdim=True)

            alpha_p, beta_p = beta_head_pref(final_scores_pos, final_scores_neg)
            var_p = compute_variance(alpha_p, beta_p)

            uncertainties_p.append(var_p.cpu().numpy())
            uncertainties_c.append(var_p.cpu().numpy())
            all_indices.append(i)

    uncertainties_p = np.concatenate(uncertainties_p)
    uncertainties_c = np.concatenate(uncertainties_c)
    all_indices = np.array(all_indices)
    return uncertainties_p, uncertainties_c, all_indices

# ---------------------------
# RewardBench Evaluation
# ---------------------------
# TODO adapt this to the beta distribution
def reward_bench_eval(args, device, gating_network, regression_model):
    gating_network.eval()
    regression_model.eval()
    print("Evaluating on RewardBench...")
    reward_bench_embeddings, reward_bench_prompt_embeddings = load_embeddings(args.reward_bench_embedding_path, device=device)

    with torch.no_grad():
        gating_weights_rb_0 = gating_network(reward_bench_prompt_embeddings[:, 0, :].squeeze())
        concept_scores_rb_0 = regression_model(reward_bench_embeddings[:, 0, :].squeeze()) 
        final_scores_rb_0 = torch.sum(concept_scores_rb_0 * gating_weights_rb_0, dim=-1)

        gating_weights_rb_1 = gating_network(reward_bench_prompt_embeddings[:, 1, :].squeeze())
        concept_scores_rb_1 = regression_model(reward_bench_embeddings[:, 1, :].squeeze()) 
        final_scores_rb_1 = torch.sum(concept_scores_rb_1 * gating_weights_rb_1, dim=-1)
        correct = (final_scores_rb_0 > final_scores_rb_1).float()

    # Load RewardBench dataset metadata
    if args.offline:
        reward_bench_ds = datasets.load_from_disk(args.path_reward_bench_data_filter)
    else:
        reward_bench_ds = datasets.load_dataset("allenai/reward-bench", split="filtered")

    df_examples = pd.DataFrame({
        "subset": reward_bench_ds["subset"],
        "correct": correct.cpu().numpy()
    })
    scores_per_section, per_subset_metrics = eval_reward_bench(df_examples)
    print("RewardBench Scores:")
    print(pd.DataFrame([scores_per_section]))
    wandb.log({"rewardbench_scores": scores_per_section})