"""Dense rank analysis for activation subspaces."""

import argparse
import json
import os

import torch
from tqdm import tqdm

from .config import POOLING, MAX_RANK, get_activation_dir, get_rank_analysis_dir, print_config, save_manifest, validate_asset


def load_activation_cache(directory):
    activations = {}
    for category in ("safe", "unsafe", "pseudo"):
        category_dir = os.path.join(directory, category)
        if not os.path.isdir(category_dir):
            continue
        for filename in sorted(os.listdir(category_dir)):
            if not filename.endswith(".pt"):
                continue
            name = filename[:-3].replace("--", ".")
            activations.setdefault(name, {})[category] = torch.load(
                os.path.join(category_dir, filename), weights_only=True
            ).float()
    return activations


def _prefix_overlap(cross_matrix, rank):
    """Mean principal-angle cosine for the first ``rank`` SVD directions."""
    return torch.linalg.svdvals(cross_matrix[:rank, :rank]).mean().item()


def _cross_matrices(decomposed):
    """Precompute pairwise products; SVD directions are already orthonormal."""
    safe_u, safe_v = decomposed["safe"]
    unsafe_u, unsafe_v = decomposed["unsafe"]
    pseudo_u, pseudo_v = decomposed["pseudo"]
    return {
        "su_u": safe_u.T @ unsafe_u,
        "sp_u": safe_u.T @ pseudo_u,
        "up_u": unsafe_u.T @ pseudo_u,
        "su_v": safe_v.T @ unsafe_v,
        "sp_v": safe_v.T @ pseudo_v,
        "up_v": unsafe_v.T @ pseudo_v,
    }


def compute_rank_overlap_dense(activations, max_rank=MAX_RANK):
    dimension = max_rank
    sums = {key: torch.zeros(dimension) for key in ("su_u", "su_v", "sp_u", "sp_v", "up_u", "up_v")}
    counts = torch.zeros(dimension)
    valid_count = 0
    for name in tqdm(sorted(activations), desc="Dense rank analysis"):
        values = activations[name]
        if not all(key in values for key in ("safe", "unsafe", "pseudo")):
            continue
        sample_sizes = {values[key].shape[0] for key in ("safe", "unsafe", "pseudo")}
        if len(sample_sizes) != 1:
            raise ValueError(
                f"Vh overlap requires equal sample counts for {name}; got {sorted(sample_sizes)}"
            )
        decomposed = {}
        usable_rank = dimension
        for category in ("safe", "unsafe", "pseudo"):
            u, _, vh = torch.linalg.svd(values[category].T, full_matrices=False)
            decomposed[category] = (u, vh.T)
            usable_rank = min(usable_rank, u.shape[1], vh.shape[0])
        if usable_rank <= 0:
            continue
        cross_matrices = _cross_matrices(decomposed)
        for rank in range(1, usable_rank + 1):
            index = rank - 1
            for key, cross_matrix in cross_matrices.items():
                sums[key][index] += _prefix_overlap(cross_matrix, rank)
        counts[:usable_rank] += 1
        valid_count += 1
    if valid_count == 0:
        raise RuntimeError("No modules contain all three activation categories")
    available = int((counts > 0).sum().item())
    return {"ranks": list(range(1, available + 1)), "valid_count": valid_count,
            "module_counts": counts[:available].int().tolist(),
            **{key: (value[:available] / counts[:available]).tolist() for key, value in sums.items()}}


def _minimum(values, ranks):
    index = min(range(len(values)), key=values.__getitem__)
    return ranks[index], values[index]


def main():
    parser = argparse.ArgumentParser(description="ProSafePrune dense rank analyzer")
    args = parser.parse_args()

    print_config()
    activation_dir = get_activation_dir()
    if not os.path.isdir(activation_dir):
        raise FileNotFoundError(activation_dir)
    validate_asset(activation_dir, "activation_cache", {"pooling": POOLING})
    output_dir = get_rank_analysis_dir()
    os.makedirs(output_dir, exist_ok=True)

    result = compute_rank_overlap_dense(load_activation_cache(activation_dir))
    rank, overlap = _minimum(result["up_v"], result["ranks"])
    candidates = [item for item, value in zip(result["ranks"], result["up_v"]) if value <= overlap + 0.005]

    with open(os.path.join(output_dir, "overlap_rank_dense.csv"), "w", encoding="utf-8") as handle:
        handle.write("rank,su_u,su_vh,sp_u,sp_vh,up_u,up_vh\n")
        for index, item in enumerate(result["ranks"]):
            handle.write(f"{item},{result['su_u'][index]:.6f},{result['su_v'][index]:.6f},"
                         f"{result['sp_u'][index]:.6f},{result['sp_v'][index]:.6f},"
                         f"{result['up_u'][index]:.6f},{result['up_v'][index]:.6f}\n")

    recommendation = {"method": "Vh unsafe-pseudo minimum", "rank": rank,
                      "overlap": overlap, "candidates": candidates,
                      "max_rank": MAX_RANK,
                      "module_counts_by_dimension": result["module_counts"],
                      "num_modules": result["valid_count"]}
    with open(os.path.join(output_dir, "rank_recommendation.json"), "w", encoding="utf-8") as handle:
        json.dump(recommendation, handle, indent=2, ensure_ascii=False)
    save_manifest(output_dir, {"asset_type": "rank_analysis", "activation_dir": activation_dir,
                               "pooling": POOLING, **recommendation})
    print(f"Recommended rank: {rank}; results: {output_dir}")


if __name__ == "__main__":
    main()
