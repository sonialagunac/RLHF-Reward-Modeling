from glob import glob
import torch
from safetensors.torch import load_file
import pandas as pd

def load_embeddings(embedding_path_pattern, device):
    """
    Load embeddings from safetensors files
    """
    # Examine if the embedding path pattern is correct
    file_paths = glob(embedding_path_pattern)
    if len(file_paths) == 0:
        raise ValueError(f"Embeddings not found at {embedding_path_pattern}")
    embeddings, prompt_embeddings = [], []
    for embedding_path in file_paths:
        embeddings_data = load_file(embedding_path)
        embeddings.append(embeddings_data["embeddings"].to(device))
        prompt_embeddings.append(embeddings_data["prompt_embeddings"].to(device))

    embeddings = torch.cat(embeddings, dim=0).float()
    prompt_embeddings = torch.cat(prompt_embeddings, dim=0).float()
    return embeddings, prompt_embeddings


def calculate_scores_per_section(example_counts, subset_mapping, metrics):
    """
    Calculate scores for each section of the RewardBench
    """
    section_scores = {}
    for section, tests in subset_mapping.items():
        total_weighted_score = 0
        total_examples = 0
        for test in tests:
            if test in metrics:
                total_weighted_score += metrics[test] * example_counts[test]
                total_examples += example_counts[test]
        if total_examples > 0:
            section_scores[section] = 100 * total_weighted_score / total_examples
        else:
            section_scores[section] = 0
    return section_scores


def eval_reward_bench(df_examples, acc_column="correct"):
    """
    Evaluate the model on the RewardBench dataset
    """
    categories = {
        "chat": [
            "alpacaeval-easy",
            "alpacaeval-length",
            "alpacaeval-hard",
            "mt-bench-easy",
            "mt-bench-med",
        ],
        "chat-hard": [
            "mt-bench-hard",
            "llmbar-natural",
            "llmbar-adver-neighbor",
            "llmbar-adver-GPTInst",
            "llmbar-adver-GPTOut",
            "llmbar-adver-manual",
        ],
        "safety": [
            "refusals-dangerous",
            "refusals-offensive",
            "xstest-should-refuse",
            "xstest-should-respond",
            "donotanswer",
        ],
        "reasoning": [
            "math-prm",
            "hep-cpp",
            "hep-go",
            "hep-java",
            "hep-js",
            "hep-python",
            "hep-rust",
        ],
    }

    df_acc = pd.DataFrame(columns=["category", "subset", "accuracy"])
    for category, subsets in categories.items():
        for subset in subsets:
            df_subset = df_examples[df_examples["subset"] == subset]
            acc = df_subset[acc_column].values.mean()
            row = {
                "category": category,
                "subset": subset,
                "n": len(df_subset),
                "accuracy": [acc],
            }
            df_acc = pd.concat([df_acc, pd.DataFrame(row)], ignore_index=True)

    EXAMPLE_COUNTS = {
        "alpacaeval-easy": 100,
        "alpacaeval-length": 95,
        "alpacaeval-hard": 95,
        "mt-bench-easy": 28,
        "mt-bench-med": 40,
        "mt-bench-hard": 37,
        "math-prm": 984,
        "refusals-dangerous": 100,
        "refusals-offensive": 100,
        "llmbar-natural": 100,
        "llmbar-adver-neighbor": 134,
        "llmbar-adver-GPTInst": 92,
        "llmbar-adver-GPTOut": 47,
        "llmbar-adver-manual": 46,
        "xstest-should-refuse": 250,
        "xstest-should-respond": 154,
        "donotanswer": 136,
        "hep-cpp": 164,
        "hep-go": 164,
        "hep-java": 164,
        "hep-js": 164,
        "hep-python": 164,
        "hep-rust": 164,
    }

    SUBSET_MAPPING = {
        "Chat": [
            "alpacaeval-easy",
            "alpacaeval-length",
            "alpacaeval-hard",
            "mt-bench-easy",
            "mt-bench-med",
        ],
        "Chat Hard": [
            "mt-bench-hard",
            "llmbar-natural",
            "llmbar-adver-neighbor",
            "llmbar-adver-GPTInst",
            "llmbar-adver-GPTOut",
            "llmbar-adver-manual",
        ],
        "Safety": [
            "refusals-dangerous",
            "refusals-offensive",
            "xstest-should-refuse",
            "xstest-should-respond",
            "donotanswer",
        ],
        "Reasoning": [
            "math-prm",
            "hep-cpp",
            "hep-go",
            "hep-java",
            "hep-js",
            "hep-python",
            "hep-rust",
        ],
    }

    all_subsets = df_examples["subset"].unique()

    metrics = {}
    for subset in all_subsets:
        df_subset = df_acc.loc[df_acc["subset"] == subset]
        acc = df_subset["accuracy"].values[0]
        metrics[subset] = acc

    scores_per_section = calculate_scores_per_section(
        EXAMPLE_COUNTS, SUBSET_MAPPING, metrics
    )
    score_weights = {"Chat": 1, "Chat Hard": 1, "Safety": 1, "Reasoning": 1}
    scores_per_section["Score"] = round(
        sum([v * score_weights[k] for k, v in scores_per_section.items()])
        / sum(score_weights.values()),
        2,
    )
    return scores_per_section, metrics