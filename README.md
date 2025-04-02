# Interpretable Rewards

## Running the model
+ **prepare_embeddings**:
+ **Run model**: train.py

## Codebase inspired on Baseline Absolute-Rating Multi-Objective Reward Model (ArmoRM) with Mixture-of-Experts (MoE) Aggregation of Reward Objectives -- Instructions to run baseline

+ **Tech Report**: https://arxiv.org/abs/2406.12845 
+ **Model**: [ArmoRM-Llama3-8B-v0.1](https://huggingface.co/RLHFlow/ArmoRM-Llama3-8B-v0.1)
  + Finetuned from model: [FsfairX-LLaMA3-RM-v0.1](https://huggingface.co/sfairXC/FsfairX-LLaMA3-RM-v0.1)
- **Code Repository:** https://github.com/RLHFlow/RLHF-Reward-Modeling/

## Training
ArmoRM has two training stages: 1) Multi-objective Reward Learning and 2) Mixture-of-Experts Gating Network Learning.

### Multi-objective Reward Learning

This stage involves training a multi-objective reward model by linear probing on top of an existing reward model. The process includes:

1. **Data Preparation** (`stage-1_prepare.py`):
   - Load a multi-objective dataset and extract embeddings from an existing reward model for each example.
   - Save embeddings and labels for further processing.
   - Example Command: 
     ```
     python stage-1_prepare.py --model_path sfairXC/FsfairX-LLaMA3-RM-v0.1 \
                               --dataset_path RLHFlow/ArmoRM-Multi-Objective-Data-v0.1 \
                               --n_shards 1 --shard_idx 1 --device 0
     ```
   - The dataset sharding (specified by `n_shards` and `shard_idx`) is optional but can be used for parallel processing.

2. **Training** (`stage-1_train.py`):
   - Perform multi-objective linear regression on the prepared embeddings, with sklearn's Ridge regression using different L2 regularization strengths for each attribute. Note: this training doesn't require a GPU.
   - Select the best regularization strength based on validation loss.
   - Save the final regression model weights in PyTorch format.
   - Example Command:
     ```
     python stage-1_train.py --model_path sfairXC/FsfairX-LLaMA3-RM-v0.1 \
                             --dataset_path RLHFlow/ArmoRM-Multi-Objective-Data-v0.1
     ```

Key aspects:
- Uses linear probing (training only the new linear layer while keeping transformer layers frozen).
- Deals with missing labels in the merged dataset by ignoring them during loss computation.

### Mixture-of-Experts Gating Network Learning

This stage involves training a gating network to aggregate the multi-objective rewards based on the context. The process includes:

1. **Data Preparation** (`stage-2_prepare.py`):
   - Load a binary preference dataset and prepare embeddings for prompts and responses using the same reward model as Stage 1.
   - Save embeddings for both chosen and rejected responses in each preference pair.
   - Example Commands:
     ```
     python stage-2_prepare.py --model_path sfairXC/FsfairX-LLaMA3-RM-v0.1 --model_family llama3 \
                               --dataset_path RLHFlow/pair_data_v2_80K_wsafety \
                               --dataset_split train --n_shards 1 --shard_idx 1 --device 0
     ```
    **Note**: 
     - You can optionally run this script with `dataset_path` referring to a reference dataset, which can be used to mitigate the verbosity bias in the following training step (e.g., `RLHFlow/UltraFeedback-preference-standard`, which is the reference dataset used in our paper). 
     - You can also run this script with `--dataset_path allenai/reward-bench --dataset_split filtered` to prepare the embeddings for the RewardBench dataset, which can be used in the following training script for evaluation.

1. **Training** (`stage-2_train.py`):
   - Verbosity Debiasing: Debiases the multi-objective rewards by applying a transformation matrix to the rewards. The coefficients of the verbosity debiasing are found by a grid search method.
   - Gating Network Learning: Train a gating network (MLP) to aggregate multi-objective rewards based on the prompt.
   - Loss Function: Use Bradley-Terry loss for training on pairwise preference data.
   - Example Command:
     ```
     python stage-2_train.py --model_path sfairXC/FsfairX-LLaMA3-RM-v0.1 --model_family llama3 \
                             --multi_objective_dataset RLHFlow/ArmoRM-Multi-Objective-Data-v0.1 \
                             --preference_dataset RLHFlow/pair_data_v2_80K_wsafety \
                             --reference_dataset RLHFlow/UltraFeedback-preference-standard \
                             --eval_reward_bench --device 0
     ```
     **Note**: 
     - If you do not specify `--reference_dataset`, the preference dataset will be used as the reference dataset for verbosity debiasing.
     - If you have run `stage-2_prepare.py` with the `--dataset_path allenai/reward-bench --dataset_split filtered, you can run this training script with `--eval_reward_bench` to evaluate the model on the RewardBench dataset. 

Key aspects:
- Freezes the backbone and regression layer from Stage 1, only training the gating network.
- Conducts verbosity debiasing on the multi-objective rewards.
- Trains the gating network with a Bradley-Terry loss.
- Evaluates performance on the RewardBench benchmark.

## Citation of the baseline
```
@article{ArmoRM,
      title={Interpretable Preferences via Multi-Objective Reward Modeling and Mixture-of-Experts}, 
      author={Haoxiang Wang and Wei Xiong and Tengyang Xie and Han Zhao and Tong Zhang},
      journal={arXiv preprint arXiv:2406.12845},
}
```
