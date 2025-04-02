from argparse import ArgumentParser
import wandb

# ---------------------------
# Set up Weights & Biases
# ---------------------------
def init_wandb(args):
    wandb.init(project="interpretable_rewards", config=vars(args))

# ---------------------------
# Arguments
# ---------------------------
def parse_args():
    parser = ArgumentParser()

    # Data and paths
    parser.add_argument("--embeddings_dir", type=str, default="path/to/hf/embeddings.safetensors",
                        help="Path pattern for embedding safetensors file(s).")
    parser.add_argument("--labels_dir", type=str, default="path/to/labels.safetensors",
                        help="Path to labels file.")
    parser.add_argument("--labels_type", type=str, default="hugging_face",
                        help="Label type (e.g. 'hugging_face' or 'model used to label the concepts').")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Directory to save outputs and model checkpoints.")

    # Training parameters
    parser.add_argument("--device", type=int, default=0, help="CUDA device id (set -1 for CPU)")
    parser.add_argument("--epochs_regression", type=int, default=1, help="Number of epochs for regression training")
    parser.add_argument("--epochs_gating", type=int, default=1, help="Number of epochs for gating network training")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate for gating network")
    parser.add_argument("--n_hidden", type=int, default=3, help="Number of hidden layers in the gating network")
    parser.add_argument("--hidden_dim", type=int, default=1024, help="Hidden dimension for gating network layers")
    parser.add_argument("--temperature", type=float, default=10, help="Temperature for softmax in gating network")
    parser.add_argument("--logit_scale", type=float, default=1.0, help="Logit scaling factor for gating network")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay for optimizers")

    # Evaluation
    parser.add_argument("--eval_reward_bench", type=bool, default=True, help="Evaluate on RewardBench after training")

    # RewardBench paths
    parser.add_argument("--reward_bench_embedding_path", type=str, default="path/to/hf/reward_bench_embedding.safetensors",
                        help="Path to RewardBench embedding file")
    parser.add_argument("--path_reward_bench_data_filter", type=str, default="path/to/hf/reward_bench_data",
                        help="Path to RewardBench dataset; for offline cluster use, specify local path")

    # Offline mode flag
    parser.add_argument("--offline", type=bool, default=True, help="If set, use offline (cluster) paths instead of downloading from HuggingFace")

return parser.parse_args()

# ---------------------------
# Update Paths for Offline Cluster
# ---------------------------
def set_offline_paths(args):
    cache_dir = "/cluster/dataset/vogtlab/Group/slaguna/huggingface/"
    args.dataset_path = os.path.join(cache_dir, "datasets", "RLHFlow___ultra_feedback-preference-standard", "default", "0.0.0", "caad75bface3d66c59a14e1d40147a8608a383b0")
    args.dataset_name = "UltraFeedback-preference-standard"
    args.model_path = os.path.join(cache_dir, "models--sfairXC--FsfairX-LLaMA3-RM-v0.1", "snapshots", "94fad49f1b3227aa8b566f415a335adb68ec544c")
    args.model_name = "FsfairX-LLaMA3-RM-v0.1"
    args.output_dir = "/cluster/dataset/vogtlab/Group/slaguna/data_RLHF/ArmoRM"
    args.embeddings_dir = os.path.join(args.output_dir, "embeddings", args.model_name, args.dataset_name + "-train.safetensors")
    args.labels_type = "hugging_face" #or "phi-3-mini-4k-instruct" or "llama"
    args.labels_dir = os.path.join(args.output_dir, "labels", args.labels_type, f"{args.dataset_name}_combined.safetensors")
    args.reward_bench_embedding_path = os.path.join(args.output_dir, "embeddings", args.model_name, "reward_bench-filtered.safetensors")
    args.path_reward_bench_data_filter = os.path.join(cache_dir, "datasets", "reward-bench-filtered")
    return args
