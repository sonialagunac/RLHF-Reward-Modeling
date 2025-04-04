from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
import numpy as np
from safetensors.torch import load_file

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

    # Concept names
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

    split = int(np.floor(0.2 * dataset_size))
    train_indices, val_indices = indices[split:], indices[:split]

    # Samplers
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    # Dataloaders
    train_dl = DataLoader(
        full_dataset,
        batch_size=cfg.model.batch_size,
        sampler=train_sampler,
        pin_memory=True,
        num_workers=cfg.get("num_workers", 0)
    )

    val_dl = DataLoader(
        full_dataset,
        batch_size=cfg.model.batch_size,
        sampler=val_sampler,
        pin_memory=True,
        num_workers=cfg.get("num_workers", 0)
    )

    return train_dl, val_dl, pos_embeddings.shape[1], concept_labels.shape[1], concepts


def sanity_check_labels(concept_labels):
    invalid_mask = (concept_labels < 0) | (concept_labels > 1)
    num_invalid = invalid_mask.sum().item()

    if num_invalid > 0:
        print(f"Found {num_invalid} label values outside [0, 1]. Setting them to 0.5 as fallback.")
        concept_labels[invalid_mask] = 0.5
    else:
        print("All labels within [0, 1].")

    return concept_labels
