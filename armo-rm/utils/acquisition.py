import numpy as np

def select_acquisition_indices(strategy, uncertainties_p, uncertainties_c, all_indices, n_samples):
    if strategy == "random":
        return acquisition_random(all_indices, n_samples)
    elif strategy == "concept_entropy":
        return acquisition_concept_entropy(uncertainties_c, all_indices, n_samples)
    else:
        raise ValueError(f"Unknown acquisition strategy: {strategy}")

def acquisition_random(all_indices, n_samples):
    return np.random.choice(all_indices, size=n_samples, replace=False)

def acquisition_concept_entropy(uncertainties_c, all_indices, n_samples):
    """
    Select top-k based on highest entropy (i.e., variance) in concept predictions
    """
    idx_sorted = np.argsort(-uncertainties_c)  # descending
    selected = all_indices[idx_sorted[:n_samples]]
    return selected
