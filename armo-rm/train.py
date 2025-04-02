# ===============================
# Unified Training Script
# Ridge Regression + Gating Network
# ===============================

import os
import torch
import numpy as np
import pandas as pd
from glob import glob
from safetensors.torch import load_file
from tqdm import tqdm
from argparse import ArgumentParser
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from networks.networks import GatingNetwork

# ---------------------------
# Arguments
# ---------------------------
parser = ArgumentParser()
parser.add_argument("--embeddings_dir", type=str)
parser.add_argument("--weights_dir", type=str)
parser.add_argument("--labels_dir_HF", type=str)
parser.add_argument("--labels_dir_LLM", type=str)
parser.add_argument("--labels_type", type=str, default='HF')
parser.add_argument("--model_label_name", type=str)
parser.add_argument("--model_path", type=str)
parser.add_argument("--dataset_path", type=str)
parser.add_argument("--output_dir", type=str)
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--n_steps", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=1024)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--n_hidden", type=int, default=3)
parser.add_argument("--hidden_dim", type=int, default=1024)
parser.add_argument("--temperature", type=float, default=10)
parser.add_argument("--eval_reward_bench", type=bool, default=False)
parser.add_argument("--logit_scale", type=float, default=1.0)
parser.add_argument("--weight_decay", type=float, default=1e-2) #using this instead of the alpha in ridge regression, consider looping through them 
parser.add_argument( "--cluster_mds",type=bool, default=True, help="Whether to use MDS Cluster, paths to models offline")
args = parser.parse_args()

if args.cluster_mds:
    cache_dir = "/cluster/dataset/vogtlab/Group/slaguna/huggingface/"
    path_UF_preference = cache_dir + "datasets/RLHFlow___ultra_feedback-preference-standard/default/0.0.0/caad75bface3d66c59a14e1d40147a8608a383b0/"
    args.dataset_path = path_UF_preference
    args.dataset_name  = "UltraFeedback-preference-standard"
    path_fsfair = cache_dir + "models--sfairXC--FsfairX-LLaMA3-RM-v0.1/snapshots/94fad49f1b3227aa8b566f415a335adb68ec544c/"
    args.model_path = path_fsfair
    args.model_name = "FsfairX-LLaMA3-RM-v0.1" 
    save_path_dir = "/cluster/dataset/vogtlab/Group/slaguna/data_RLHF/ArmoRM"
    args.output_dir = save_path_dir
    args.embeddings_dir = os.path.join(save_path_dir + "/embeddings/", args.model_name, args.dataset_name + "-train.safetensors")
    args.weights_dir = save_path_dir + "/regression_weights/"
    args.model_label_name = "phi-3-mini-4k-instruct"
    args.labels_dir_LLM = save_path_dir + f"/labels/{args.model_label_name}/{args.dataset_name}_combined.safetensors"
    args.labels_dir_HF = cache_dir + "ultrafeedback_preference_with_annotations"
device = f"cuda:{args.device}" if args.device >= 0 else "cpu"
concepts = ["helpfulness", "correctness", "coherence", "complexity", "verbosity", "overall_score", "instruction_following", "truthfulness", "honesty",  "is_safe", "score", "overall_quality", "judge_lm", "style", "explanation", "readability"]

# ---------------------------
# Load concept labels + embeddings from safetensors
# ---------------------------
print("Loading embeddings...")
embed_data = load_file( args.embeddings_dir )
embeddings = embed_data["embeddings"].float()
prompt_embeddings = embed_data["prompt_embeddings"].float()

if args.labels_type == 'HF':
    label_data = load_file(args.labels_dir_LLM)
    concept_labels = label_data["concepts_label"].float()
else:
    ds = datasets.load_dataset(args.labels_dir_HF, split=args.dataset_split)
    
# Get pairwise 
pos_embeddings = embeddings[:, 0]       
neg_embeddings = embeddings[:, 1] 

pos_prompt_embeddings = prompt_embeddings[:, 0]   
neg_prompt_embeddings = prompt_embeddings[:, 1] 

#-----
# Temporary Sonia
noise = np.random.uniform(-1e-1, 1e-1, size=(concept_labels.shape[0],concept_labels.shape[-1] ))
concept_labels[:, 1]= np.clip(concept_labels[:, 1] + noise, 0.0, 1.0)
# ------
label_diffs = concept_labels[:, 0] - concept_labels[:, 1] 

print("Embeddings:", embeddings.shape)
print("Concept label diffs:", label_diffs.shape)


# ---------------------------
# Train/Val split
# ---------------------------
X_pos_train, X_pos_val, X_neg_train, X_neg_val, X_pos_prompt_train, X_pos_prompt_val, X_neg_prompt_train, X_neg_prompt_val,  y_train, y_val = train_test_split(
    pos_embeddings, neg_embeddings, pos_prompt_embeddings, neg_prompt_embeddings, label_diffs, test_size=0.2, random_state=42
)

# ---------------------------
# Define linear regression model
# ---------------------------
input_dim = pos_embeddings.shape[1]  # 4096
output_dim = label_diffs.shape[1]    # 17

regression_model = nn.Linear(input_dim, output_dim, bias=False).to(device)
optimizer = torch.optim.AdamW(regression_model.parameters(), lr=1e-3, weight_decay=args.weight_decay)
loss_fn = nn.MSELoss()

# ---------------------------
# Define gating model
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
scheduler_gate = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_gate, T_max=args.n_steps)
loss_gate_fn = nn.BCEWithLogitsLoss()

# ---------------------------
# Create DataLoaders
# ---------------------------
train_ds = TensorDataset(X_pos_train, X_neg_train, X_pos_prompt_train, X_neg_prompt_train, y_train)
val_ds = TensorDataset(X_pos_val, X_neg_val, X_pos_prompt_val, X_neg_prompt_val, y_val)

train_dl = DataLoader(train_ds, batch_size=1024, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=1024)

# ---------------------------
# Training Regression Loop
# ---------------------------
print("Training multivariate regression model on concept score deltas...")
for epoch in range(1):
    regression_model.train()
    total_loss = 0
    for pos, neg, _, _, y in train_dl:
        pos, neg, y = pos.to(device), neg.to(device), y.to(device)
        pred_pos = regression_model(pos)  # [B, 17]
        pred_neg = regression_model(neg)  # [B, 17]
        pred_diff = pred_pos - pred_neg

        loss = loss_fn(pred_diff, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1} - Train Loss: {total_loss/len(train_dl):.4f}")

# ---------------------------
# Final Validation Loss
# ---------------------------
regression_model.eval()
with torch.no_grad():
    val_loss = 0
    for pos, neg, _, _, y in val_dl:
        pos, neg, y = pos.to(device), neg.to(device), y.to(device)
        pred_diff = regression_model(pos) - regression_model(neg)
        val_loss += loss_fn(pred_diff, y).item()
    print(f"Final Validation Loss: {val_loss / len(val_dl):.4f}")

# ---------------------------
# Save Regression Weights
# ---------------------------
regression_weights = regression_model.weight.detach().cpu()  
torch.save({"weight": regression_weights}, os.path.join(args.weights_dir + args.model_name + "_" + args.dataset_name + ".pt"))
print(f"Saved regression weights to {args.weights_dir }{args.model_name}_{args.dataset_name}.pt")


# ---------------------------
# Training Gating Loop
# ---------------------------
regression_model.eval()
for step in tqdm(range(args.n_steps)):
    for pos, neg, pos_prompt, neg_prompt, _ in train_dl:
        pos, neg, pos_prompt, neg_prompt = pos.to(device), neg.to(device), pos_prompt.to(device), neg_prompt.to(device)

        weights = gating_network(pos_prompt)      
        concept_scores = regression_model(pos) 
        final_scores_pos = torch.sum(concept_scores * weights, dim=-1) 

        weights_neg = gating_network(neg_prompt)      
        concept_scores_neg = regression_model(neg) 
        final_scores_neg = torch.sum(concept_scores_neg * weights_neg, dim=-1) 
        loss_gate = loss_gate_fn(final_scores_pos - final_scores_neg, torch.ones_like(final_scores_pos))

        optimizer_gate.zero_grad()
        loss_gate.backward()
        optimizer_gate.step()
        scheduler_gate.step()

        if step % 100 == 0:
            print(f"Step {step} - Loss: {loss_gate.item():.4f}")

# Evaluation
gating_network.eval()
with torch.no_grad():
    val_acc = 0
    for pos, neg, pos_prompt, neg_prompt, _ in val_dl:
        weights_val_pos = gating_network(pos_prompt)
        weights_val_neg = gating_network(neg_prompt)
        concept_scores = regression_model(pos) 
        concept_scores_neg = regression_model(neg) 

        final_scores_pos = torch.sum(concept_scores * weights, dim=-1) 
        final_scores_neg = torch.sum(concept_scores_neg * weights_neg, dim=-1) 

        acc = ((final_scores_pos - final_scores_neg) > 0).float().mean()
        val_acc += acc
    print(f"Final Validation Accuracy: {acc / len(val_dl):.4f}")

# Save
save_path = os.path.join(args.output_dir, f"gating_network_{args.model_name}_{args.dataset_name}.pt")
torch.save(gating_network.state_dict(), save_path)
print(f"Gating Network saved to {save_path}")


# ==========================
# RewardBench Evaluation
# ==========================
if args.eval_reward_bench:
    print("Evaluating on RewardBench...")

    # Load RewardBench embeddings
    reward_bench_embeddings, reward_bench_prompt_embeddings = load_embeddings(
        args.reward_bench_embedding_path, device=device
    )

    with torch.no_grad():
        gating_weights_rb = gating_network(reward_bench_prompt_embeddings)
        concept_scores_rb = reward_bench_embeddings @ regression_weights.T
        final_scores_rb = torch.sum(concept_scores_rb * gating_weights_rb, dim=-1)
        correct = (final_scores_rb[:, 0] > final_scores_rb[:, 1]).float()

    # Load reward bench dataset metadata
    if args.cluster_mds:
        reward_bench_ds = load_from_disk("/cluster/dataset/vogtlab/Group/slaguna/huggingface/datasets/reward-bench-filtered/")
    else:
        reward_bench_ds = load_dataset("allenai/reward-bench", split="filtered")

    # Accuracy per subset
    df_examples = pd.DataFrame({
        "subset": reward_bench_ds["subset"],
        "correct": correct.cpu().numpy()
    })

    def eval_reward_bench(df_examples):
        categories = {
            "chat": ["alpacaeval-easy", "alpacaeval-length", "alpacaeval-hard", "mt-bench-easy", "mt-bench-med"],
            "chat-hard": ["mt-bench-hard", "llmbar-natural", "llmbar-adver-neighbor", "llmbar-adver-GPTInst", "llmbar-adver-GPTOut", "llmbar-adver-manual"],
            "safety": ["refusals-dangerous", "refusals-offensive", "xstest-should-refuse", "xstest-should-respond", "donotanswer"],
            "reasoning": ["math-prm", "hep-cpp", "hep-go", "hep-java", "hep-js", "hep-python", "hep-rust"]
        }
        EXAMPLE_COUNTS = {
            "alpacaeval-easy": 100, "alpacaeval-length": 95, "alpacaeval-hard": 95, "mt-bench-easy": 28, "mt-bench-med": 40,
            "mt-bench-hard": 37, "llmbar-natural": 100, "llmbar-adver-neighbor": 134, "llmbar-adver-GPTInst": 92,
            "llmbar-adver-GPTOut": 47, "llmbar-adver-manual": 46, "refusals-dangerous": 100, "refusals-offensive": 100,
            "xstest-should-refuse": 250, "xstest-should-respond": 154, "donotanswer": 136, "math-prm": 984,
            "hep-cpp": 164, "hep-go": 164, "hep-java": 164, "hep-js": 164, "hep-python": 164, "hep-rust": 164,
        }

        def section_scores(metrics):
            scores = {}
            for sec, subsets in categories.items():
                total, count = 0, 0
                for s in subsets:
                    acc = metrics.get(s, 0)
                    n = EXAMPLE_COUNTS.get(s, 0)
                    total += acc * n
                    count += n
                scores[sec] = 100 * total / count if count else 0
            return scores

        metrics = df_examples.groupby("subset")["correct"].mean().to_dict()
        section = section_scores(metrics)
        section["Score"] = round(np.mean(list(section.values())), 2)
        return section, metrics

    scores_per_section, per_subset_metrics = eval_reward_bench(df_examples)
    print("RewardBench Scores:")
    print(pd.DataFrame([scores_per_section]))