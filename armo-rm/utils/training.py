from utils.utils import load_embeddings, eval_reward_bench
import pandas as pd
import wandb, datasets, torch


# --------------------------------------
# Training and Eval Regression Functions
# --------------------------------------
def train_regression(model, optimizer, loss_fn, dataloader, device, epoch, total_epochs):
    model.train()
    total_loss = 0
    for pos, neg, _, _, y in dataloader:
        pos, neg, y = pos.to(device), neg.to(device), y.to(device)
        optimizer.zero_grad()
        pred_diff = model(pos) - model(neg)
        loss = loss_fn(pred_diff, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch}/{total_epochs} - Regression Train Loss: {avg_loss:.4f}")
    wandb.log({"regression_train_loss": avg_loss, "epoch": epoch})
    return avg_loss


def validate_regression(model, loss_fn, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for pos, neg, _, _, y in dataloader:
            pos, neg, y = pos.to(device), neg.to(device), y.to(device)
            pred_diff = model(pos) - model(neg)
            loss = loss_fn(pred_diff, y)
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    print(f"Regression Validation Loss: {avg_loss:.4f}")
    wandb.log({"regression_val_loss": avg_loss})
    return avg_loss

# ----------------------------------
# Training and Eval Gating Functions
# ----------------------------------
def train_gating(gating_network, regression_model, optimizer_gate, loss_gate_fn, scheduler_gate, dataloader, device, epoch):
    gating_network.train()
    regression_model.eval()  # keep regression model fixed during gating training
    total_loss = 0
    for pos, neg, pos_prompt, neg_prompt, _ in dataloader:
        pos, neg, pos_prompt, neg_prompt = pos.to(device), neg.to(device), pos_prompt.to(device), neg_prompt.to(device)
        optimizer_gate.zero_grad()
        
        # Forward pass for positive samples
        weights_pos = gating_network(pos_prompt)
        concept_scores_pos = regression_model(pos)
        final_scores_pos = torch.sum(concept_scores_pos * weights_pos, dim=-1)
        
        # Forward pass for negative samples
        weights_neg = gating_network(neg_prompt)
        concept_scores_neg = regression_model(neg)
        final_scores_neg = torch.sum(concept_scores_neg * weights_neg, dim=-1)
        
        loss_gate = loss_gate_fn(final_scores_pos - final_scores_neg, torch.ones_like(final_scores_pos))
        loss_gate.backward()
        optimizer_gate.step()
        scheduler_gate.step()
        
        total_loss += loss_gate.item()
    avg_loss = total_loss / len(dataloader)
    print(f"Gating Training Step {epoch} - Loss: {avg_loss:.4f}")
    wandb.log({"gating_train_loss": avg_loss, "gating_epoch": epoch})
    return avg_loss

def validate_gating(gating_network, regression_model, val_dl, device):
    gating_network.eval()
    regression_model.eval()
    with torch.no_grad():
        val_acc = 0
        for pos, neg, pos_prompt, neg_prompt, _ in val_dl:
            pos, neg, pos_prompt, neg_prompt = pos.to(device), neg.to(device), pos_prompt.to(device), neg_prompt.to(device)
            weights_val_pos = gating_network(pos_prompt)
            weights_val_neg = gating_network(neg_prompt)
            concept_scores_pos = regression_model(pos) 
            concept_scores_neg = regression_model(neg)
            final_scores_pos = torch.sum(concept_scores_pos * weights_val_pos, dim=-1) 
            final_scores_neg = torch.sum(concept_scores_neg * weights_val_neg, dim=-1)
            acc = ((final_scores_pos - final_scores_neg) > 0).float().mean()
            val_acc += acc.item()
        avg_acc = val_acc / len(val_dl)
        print(f"Final Validation Accuracy (Gating): {avg_acc:.4f}")
        wandb.log({"gating_val_accuracy": avg_acc})


# ---------------------------
# RewardBench Evaluation
# ---------------------------
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