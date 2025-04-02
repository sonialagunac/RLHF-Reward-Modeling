import torch
import glob
import os
from safetensors.torch import load_file, save_file

# Parameters
cache_dir = "/cluster/dataset/vogtlab/Group/slaguna/huggingface/"
save_path_dir = "/cluster/dataset/vogtlab/Group/slaguna/"
dataset_name = "UltraFeedback-preference-standard"
# model_name = "mistral-7b-instruct"
model_name = "phi-3-mini-4k-instruct"

save_path =  save_path_dir + f"data_RLHF/ArmoRM/labels/{model_name}/{dataset_name}"
num_batches = 20 #10

# Load batch files
batch_files = [f"{save_path}_batch_{i}.safetensors" for i in range(num_batches)]

# Load tensors from all batches
tensors = []
for file in batch_files:
    if os.path.exists(file):
        data = load_file(file)["concepts_label"]
        tensors.append(data)
    else:
        raise FileNotFoundError(f"Batch file {file} not found.")

# Concatenate tensors along the batch dimension
combined_tensor = torch.cat(tensors, dim=1)

# Save combined tensor
combined_save_path = f"{save_path}_combined.safetensors"
save_file({"concepts_label": combined_tensor}, combined_save_path)

print(f"Combined labels saved to {combined_save_path}")

