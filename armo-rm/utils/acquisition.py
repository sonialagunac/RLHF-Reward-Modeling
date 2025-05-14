import numpy as np

def select_acquisition_indices(strategy, means_c, all_indices, n_samples, n_concepts):
    if strategy == "random":
        return acquisition_random(means_c, all_indices, n_samples, n_concepts )
    elif strategy == "concept_entropy":
        return acquisition_concept_entropy(means_c, all_indices, n_samples)
    elif strategy == "high_confusion":
        return acquisition_high_confusion(means_c, all_indices, n_samples, n_concepts)
    else:
        raise ValueError(f"Unknown acquisition strategy: {strategy}")

def acquisition_random(means_c, all_indices, n_samples, n_concepts):
    sample_idxs = np.random.choice(all_indices, size=n_samples, replace=False)  # shape: (n_samples, n_concepts)
    concept_idxs = np.array([
        np.random.choice(np.arange(means_c.shape[-1]), size=n_concepts, replace=False)
        for _ in range(n_samples)
    ]) 
    return [(sample_idx, concept_row.tolist()) for sample_idx, concept_row in zip(sample_idxs, concept_idxs)]

def acquisition_concept_entropy(uncertainties_c, all_indices, n_samples):
    """
    Select top-k based on highest entropy (i.e., variance) in concept predictions
    """
    idx_sorted = np.argsort(-uncertainties_c)  # descending
    selected = all_indices[idx_sorted[:n_samples]]
    return selected

def acquisition_high_confusion(means, all_indices, n_samples, n_concepts):
    """
    Approximate uncertainty with proximity to 0.5.
    """
    uncertainty = 1.0 - np.abs(means - 0.5) * 2  # Shape: (N, C)

    total_uncertainty = uncertainty.sum(axis=1)
    top_sample_idxs = np.argsort(-total_uncertainty)[:n_samples]

    concept_idxs = np.argsort(-uncertainty[top_sample_idxs], axis=1)[:, :n_concepts]

    return [(int(all_indices[sample_idx]), concept_row.tolist())
            for sample_idx, concept_row in zip(top_sample_idxs, concept_idxs)]
