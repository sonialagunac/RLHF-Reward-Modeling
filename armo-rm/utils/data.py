from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
import numpy as np
from safetensors.torch import load_file


# Dataloaders
def make_loader(cfg, full_dataset, sampler):
    return DataLoader(
        full_dataset,
        batch_size=cfg.model.batch_size,
        sampler=sampler,
        pin_memory=True,
        num_workers=cfg.get("num_workers", 0)
    )

def get_dataloaders(cfg):
    print("Loading embeddings and labels...")

    # Load embeddings
    embed_data = load_file(cfg.data.embeddings_dir)
    embeddings = embed_data["embeddings"].float()
    prompt_embeddings = embed_data["prompt_embeddings"].float()

    # Load concept labels
    label_data = load_file(cfg.data.labels_dir)
    concept_labels = label_data["concepts_label"].float().transpose(0, 1)

    # Optionally fix label ranges
    if cfg.data.sanity_check_labels:
        concept_labels = sanity_check_labels(concept_labels)

    # Concept names # TODO remove this and instead load them from some file depending on the label type once we have a lot of them
    if cfg.data.labels_type == 'hugging_face':
        concepts = ["helpfulness", "honesty", "instruction_following", "truthfulness"]
    else:
        concepts = ["helpfulness", "correctness", "coherence", "complexity", "verbosity",
                    "overall_score", "instruction_following", "truthfulness", "honesty",  
                    "is_safe", "score", "overall_quality", "judge_lm", "style", "explanation", "readability"]

    # Unpack pairwise embeddings
    pos_embeddings = embeddings[:, 0]
    neg_embeddings = embeddings[:, 1]
    pos_prompt_embeddings = prompt_embeddings[:, 0]
    neg_prompt_embeddings = prompt_embeddings[:, 1]

    print("Embeddings shape:", embeddings.shape)
    print("Concept labels shape:", concept_labels.shape)

    # Create full dataset
    full_dataset = TensorDataset(pos_embeddings, neg_embeddings, pos_prompt_embeddings, neg_prompt_embeddings, concept_labels)

    # Train/val split
    dataset_size = len(full_dataset)
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)

    # Get split ratios from config
    split_ratio = cfg.data.split_ratio  # [train, val, test]
    assert abs(sum(split_ratio) - 1.0) < 1e-5, "Split ratios must sum to 1.0"
    train_end = int(split_ratio[0] * dataset_size)
    val_end = train_end + int(split_ratio[1] * dataset_size)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    # Samplers
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_dl = make_loader(cfg, full_dataset, train_sampler)
    val_dl = make_loader(cfg, full_dataset, val_sampler)
    test_dl = make_loader(cfg, full_dataset, test_sampler)

    return train_dl, val_dl, test_dl, pos_embeddings.shape[1], concept_labels.shape[1], concepts

    
def sanity_check_labels(concept_labels):
    invalid_mask = (concept_labels < 0) | (concept_labels > 1)
    num_invalid = invalid_mask.sum().item()

    if num_invalid > 0:
        print(f"Found {num_invalid} label values outside [0, 1]. Setting them to 0.5 as fallback.")
        concept_labels[invalid_mask] = 0.5
    else:
        print("All labels within [0, 1].")

    return concept_labels
