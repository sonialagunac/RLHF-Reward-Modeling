import os
import torch
import datasets
from transformers import AutoTokenizer, AutoModel, AutoConfig
from tqdm.auto import tqdm
from safetensors.torch import save_file
from argparse import ArgumentParser

# Set up CUDA optimizations for faster computation
torch.backends.cuda.matmul.allow_tf32 = (
    True  # Enable TensorFloat-32 matrix multiplication on CUDA
)
torch.backends.cudnn.allow_tf32 = (
    True  # Allow TensorFloat-32 in cuDNN for faster convolution operations
)

# Define token patterns for gating different model families
token_patterns = {
    # Llama3 token IDs of "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    "llama3": [128009, 128006, 78191, 128007, 271],
    # Gemma2 token IDs of "<end_of_turn>\n<start_of_turn>model\n"
    "gemma2": [107, 108, 106, 2516, 108],
}


def find_token_for_gating(lst, model_family):
    """Find the last occurrence of a token_pattern in a list."""
    token_pattern = token_patterns[model_family]
    token_pattern_len = len(token_pattern)
    search_end = len(lst)
    for j in range(search_end - token_pattern_len, -1, -1):
        if lst[j : j + token_pattern_len] == token_pattern:
            return j
    raise ValueError("Token pattern not found in the list.")

# Initialize the argument parser to handle command-line inputs
parser = ArgumentParser()
parser.add_argument(
    "--model_path",
    type=str,
    default="sfairXC/FsfairX-LLaMA3-RM-v0.1",
    help="Path to the pre-trained model (HuggingFace path or local folder)",
)
parser.add_argument(
    "--model_family", type=str, default="llama3", help="Model family (llama3 or gemma2)"
)
parser.add_argument(
    "--dataset_path",
    type=str,
    default="RLHFlow/UltraFeedback-preference-standard", #note the demo mentions this one instead --dataset_path RLHFlow/pair_data_v2_80K_wsafety
    help="Path to the dataset (HuggingFace path or local folder)",
)
parser.add_argument(
    "--source", default=None, type=str, help="Source filter for the dataset"
)
parser.add_argument(
    "--dataset_split", type=str, default="train", help="Dataset split to use"
)
parser.add_argument(
    "--n_shards",
    type=int,
    default=1,
    help="Total number of shards to divide the dataset into",
)
parser.add_argument(
    "--shard_idx", type=int, default=1, help="Index of the current shard"
)
parser.add_argument(
    "--device", type=int, default=0, help="CUDA device index to use for computation"
)
parser.add_argument(
    "--seq_len", type=int, default=8192, help="Maximum sequence length for input"
)

parser.add_argument(
    "--cluster_mds",
    type=bool,
    default=True,
    help="Whether to use MDS Cluster, paths to models offline",
)
parser.add_argument(
    "--UF",
    type=bool,
    default=False,
    help="Whether to use MDS Cluster and UF dataset",
)
parser.add_argument(
    "--RB",
    type=bool,
    default=False,
    help="Whether to use MDS Cluster and reward_bench dataset",
)

args = parser.parse_args()  # Parse the provided command-line arguments

# Set up paths for saving embeddings
HOME = os.path.expanduser("~")
model_name = args.model_path.split("/")[-1]
dataset_name =  args.dataset_path.split("/")[-1]
save_path = HOME + f"/data/ArmoRM/embeddings_prompt/{model_name}/{dataset_name}"

if args.cluster_mds:
    cache_dir = "/cluster/dataset/vogtlab/Group/slaguna/huggingface/"
    path_pref_data= cache_dir + "datasets/RLHFlow___pair_data_v2_80_k_wsafety/default/0.0.0/8115933aa14a9d1c06fc9eb96311df38bbf65419/"
    path_fsfair = cache_dir + "models--sfairXC--FsfairX-LLaMA3-RM-v0.1/snapshots/94fad49f1b3227aa8b566f415a335adb68ec544c/"
    save_path_dir = "/cluster/dataset/vogtlab/Group/slaguna/"
    model_name = "FsfairX-LLaMA3-RM-v0.1" 
    dataset_name = "pair_data_v2_80K_wsafety" 
    save_path =  save_path_dir + f"data_RLHF/ArmoRM/embeddings_prompt/{model_name}/{dataset_name}"
    args.dataset_path = path_pref_data
    if args.UF:
        path_UF_preference = cache_dir + "datasets/RLHFlow___ultra_feedback-preference-standard/default/0.0.0/caad75bface3d66c59a14e1d40147a8608a383b0/"
        args.dataset_path = path_UF_preference
        dataset_name = "UltraFeedback-preference-standard"
    elif args.RB:
        path_reward_bench_data_filter = cache_dir + "datasets/reward-bench-filtered/"
        args.dataset_path = path_reward_bench_data_filter
        dataset_name = "reward_bench"
    args.model_path = path_fsfair
    

if args.source is not None:
    save_path += f"-{args.source}"
save_path += f"-{args.dataset_split}"

# Verify that the model family is passed correctly
config = AutoConfig.from_pretrained(args.model_path)
if args.model_family == "llama3":
    assert config.model_type == "llama"
elif args.model_family == "gemma2":
    assert config.model_type == "gemma2"
else:
    raise ValueError(f"Model family {args.model_family} is not supported")


# Load and prepare the dataset
ds = datasets.load_dataset(args.dataset_path, split=args.dataset_split)
if args.source is not None:
    ds = ds.filter(lambda x: x["source"] == args.source)
if args.n_shards > 1:
    ds = ds.shuffle(seed=0)
    ds = ds.shard(num_shards=args.n_shards, index=args.shard_idx - 1)

# Load the pre-trained model and tokenizer
device = f"cuda:{args.device}"
model = AutoModel.from_pretrained(
    args.model_path,
    torch_dtype=torch.bfloat16,  # Use bfloat16 precision for model weights to save memory
    device_map=device,
    # attn_implementation="flash_attention_2",  # Specify the attention implementation for efficiency
)
tokenizer = AutoTokenizer.from_pretrained(args.model_path)

# Initialize lists to store embeddings and prompt embeddings
embeddings = []
prompt_embeddings = []

# Process each example in the dataset
for example in tqdm(ds, desc="Examples"):
    chosen = example["chosen"]
    rejected = example["rejected"]

    if "prompt" in example:
        # Format the data with the standard chat template if prompt is available
        chosen = [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": chosen},
        ]
        rejected = [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": rejected},
        ]

    pair_embeddings = []
    pair_prompt_embeddings = []

    for iter_example in [chosen, rejected]:
        # Format the conversation messages using the tokenizer's chat template without tokenization
        # Extract only the user message (first 'role': 'user' entry)
        user_prompt = next((msg["content"] for msg in iter_example if msg["role"] == "user"), None)
        # Tokenize the formatted conversation and move tensors to the specified device
        conv_tokenized = tokenizer(user_prompt, return_tensors="pt").to(device)

        input_ids = conv_tokenized["input_ids"]

        # We only have one sequence so batch size is 1
        if input_ids.shape[1] > args.seq_len:
            continue

        with torch.no_grad():
            output = model(**conv_tokenized)
            last_hidden_state = output.last_hidden_state[0]
            prompt_embedding = last_hidden_state[-1].cpu()
            pair_prompt_embeddings.append(prompt_embedding)

    # Only add the pair if both chosen and rejected embeddings were successfully computed
    if len(pair_prompt_embeddings) == 2:
        prompt_embeddings.append(torch.stack(pair_prompt_embeddings))

# Convert lists of embeddings to tensors
prompt_embeddings = torch.stack(prompt_embeddings)

# Prepare the directory for saving embeddings
os.makedirs(os.path.dirname(save_path), exist_ok=True)
file_name = (
    f"{save_path}-{args.shard_idx:05d}-of-{args.n_shards:05d}"
    if args.n_shards > 1
    else save_path
)

# Save the embeddings and prompt embeddings using safetensors
save_file(
    {"prompt_embeddings": prompt_embeddings},
    f"{save_path}.safetensors",
)

# Print a confirmation message with the path to the saved embeddings
print(f"Saved embeddings to {save_path}.safetensors")
