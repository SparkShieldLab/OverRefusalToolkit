"""Build low-rank ProSafePrune bases from cached activations.

The builder applies thin SVD to every category and module, then stores basis
vectors and singular values up to the configured maximum rank.
"""

import os, json, argparse, time
import torch
from tqdm import tqdm

from .config import (
    MODEL_ID, POOLING, MAX_RANK,
    get_activation_dir, get_subspace_dir,
    save_manifest, print_config, validate_asset,
)


def load_activation_cache(act_dir):
    """Load cached activations from disk."""
    activations = {}
    for category in ["safe", "unsafe", "pseudo"]:
        cat_dir = os.path.join(act_dir, category)
        if not os.path.exists(cat_dir):
            continue
        for fname in sorted(os.listdir(cat_dir)):
            if not fname.endswith('.pt'):
                continue
            layer_name = fname.replace('.pt', '').replace('--', '.')
            tensor = torch.load(os.path.join(cat_dir, fname), weights_only=True)
            if layer_name not in activations:
                activations[layer_name] = {}
            activations[layer_name][category] = tensor.float()
    return activations


def compute_basis(activations, max_rank=MAX_RANK):
    """Compute category-specific bases and singular values for each module."""
    bases_safe, bases_unsafe, bases_pseudo = {}, {}, {}
    singular_values = {}

    for name in tqdm(sorted(activations.keys()), desc="  SVD"):
        acts = activations[name]
        for cat, storage in [("safe", bases_safe), ("unsafe", bases_unsafe),
                              ("pseudo", bases_pseudo)]:
            if cat not in acts or acts[cat].numel() == 0:
                continue
            U, S, Vh = torch.linalg.svd(acts[cat].T, full_matrices=False)
            actual_rank = min(max_rank, U.shape[1])
            storage[name] = U[:, :actual_rank]
            if name not in singular_values:
                singular_values[name] = {}
            singular_values[name][cat] = S[:actual_rank]

    return bases_safe, bases_unsafe, bases_pseudo, singular_values


def main():
    parser = argparse.ArgumentParser(description="ProSafePrune basis builder")
    args = parser.parse_args()

    print_config()

    act_dir = get_activation_dir()

    if not os.path.exists(act_dir):
        print(f"Error: activation cache not found: {act_dir}")
        raise FileNotFoundError(act_dir)

    activation_manifest = validate_asset(act_dir, "activation_cache", {"pooling": POOLING})
    out_dir = get_subspace_dir()
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n[1/3] Loading activations from: {act_dir}")
    activations = load_activation_cache(act_dir)
    print(f"  Loaded {len(activations)} modules")

    # SVD
    print(f"\n[2/3] Computing thin SVD subspaces (max rank={MAX_RANK})...")
    bases_safe, bases_unsafe, bases_pseudo, svals = compute_basis(activations)

    print(f"\n[3/3] Saving basis...")
    torch.save(bases_safe, os.path.join(out_dir, "safe_basis.pt"))
    torch.save(bases_unsafe, os.path.join(out_dir, "unsafe_basis.pt"))
    torch.save(bases_pseudo, os.path.join(out_dir, "pseudo_basis.pt"))
    torch.save(svals, os.path.join(out_dir, "singular_values.pt"))

    save_manifest(out_dir, {
        "asset_type": "subspaces",
        "activation_dir": act_dir,
        "pooling": activation_manifest["pooling"],
        "max_rank": MAX_RANK,
        "activation_manifest_sha256": activation_manifest.get("params_sha256"),
        "num_layers": len(bases_safe),
        "example_shape": list(next(iter(bases_safe.values())).shape) if bases_safe else None,
    })

    print(f"\nSubspaces built ({len(bases_safe)} modules).")
    print(f"   Subspaces: {out_dir}")
    print(f"   To use a specific rank: basis[:, :rank]")


if __name__ == "__main__":
    main()
