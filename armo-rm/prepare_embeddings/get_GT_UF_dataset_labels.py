import os
import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm
import datasets

# ---------------------------
# Arguments
# ---------------------------
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--device", type=int, default=0)
args = parser.parse_args()

# ---------------------------
# Paths and Config
# ---------------------------
cache_dir = "/cluster/dataset/vogtlab/Group/slaguna/huggingface/"
dataset_name = "UltraFeedback-preference-standard"
model_name = "hugging_face"
save_path_dir = "/cluster/dataset/vogtlab/Group/slaguna/data_RLHF/ArmoRM"
labels_dir_HF = os.path.join(cache_dir, "ultrafeedback_preference_with_annotations")
save_path = os.path.join(save_path_dir, f"labels/{model_name}/{dataset_name}")
device = f"cuda:{args.device}" if args.device >= 0 else "cpu"
max_diff = 4.0  # static max difference in the UF fine grained scores
# ---------------------------
# Processing
# ---------------------------
ds = datasets.load_from_disk(labels_dir_HF)
scores = []
for example in tqdm(ds, desc="Examples"):
    c_score_tensor = torch.tensor(example['chosen_ratings'], dtype=torch.float32)
    r_score_tensor = torch.tensor(example["rejected_ratings"], dtype=torch.float32)
    scores.append((c_score_tensor - r_score_tensor) / max_diff)
combined_tensor = torch.stack(scores, dim=1)

combined_save_path = f"{save_path}_combined.safetensors"
save_file({"concepts_label": combined_tensor}, combined_save_path)

print(f"Combined labels saved to {combined_save_path}")
