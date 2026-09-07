"""Analyze layer separability from cached ProSafePrune activations.

The analysis computes three-class silhouette scores for each module, aggregates
them by layer, and writes candidate layers and detailed scores to CSV and JSON.
"""

import os, json, argparse, time
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import silhouette_score

from .config import (
    MODEL_ID, TARGET_MODULES, POOLING,
    get_activation_dir, get_layer_analysis_dir,
    save_manifest, print_config, parse_layer_id, validate_asset,
)


def load_activation_cache(act_dir):
    """Load activations as ``{module: {category: tensor[N, D]}}``."""
    activations = {}
    for category in ["safe", "unsafe", "pseudo"]:
        cat_dir = os.path.join(act_dir, category)
        if not os.path.exists(cat_dir):
            print(f"  Warning: missing activation directory: {cat_dir}")
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


def compute_module_silhouettes(activations):
    """Compute a three-class silhouette score for each module."""
    scores = {}
    for name in tqdm(sorted(activations.keys()), desc="  Computing silhouette"):
        acts = activations[name]
        if 'safe' not in acts or 'unsafe' not in acts or 'pseudo' not in acts:
            continue
        if acts['safe'].numel() == 0 or acts['unsafe'].numel() == 0 or acts['pseudo'].numel() == 0:
            continue

        X = torch.cat([acts['safe'], acts['unsafe'], acts['pseudo']], dim=0).numpy()
        n_s, n_u, n_p = acts['safe'].shape[0], acts['unsafe'].shape[0], acts['pseudo'].shape[0]
        labels = np.array([0]*n_s + [1]*n_u + [2]*n_p)

        try:
            score = silhouette_score(X, labels, random_state=42)
        except Exception:
            score = -1.0
        scores[name] = float(score)

    return scores


def aggregate_to_layers(module_scores, method="mean"):
    """Aggregate module scores by layer."""
    layer_scores = {}
    for name, score in module_scores.items():
        layer_id = parse_layer_id(name)
        if layer_id is None:
            continue

        if layer_id not in layer_scores:
            layer_scores[layer_id] = []
        layer_scores[layer_id].append(score)

    if method == "mean":
        return {lid: float(np.mean(scores)) for lid, scores in layer_scores.items()}
    elif method == "median":
        return {lid: float(np.median(scores)) for lid, scores in layer_scores.items()}
    else:
        raise ValueError(f"Unknown aggregation: {method}")


def find_candidate_layers(layer_scores, num_layers, window=5, middle_ratio=0.3):
    """Select a candidate window around the best-scoring middle layer.

    Args:
        layer_scores: {layer_id: float}
        num_layers: Total number of model layers.
        window: Candidate window size.
        middle_ratio: Fraction excluded from both ends of the model.

    Returns:
        dict with best_layer, candidate_layers, etc.
    """
    bottom = int(num_layers * middle_ratio)
    top = int(num_layers * (1 - middle_ratio))
    middle_scores = {lid: s for lid, s in layer_scores.items() if bottom <= lid <= top}

    if not middle_scores:
        middle_scores = layer_scores  # fallback

    best_layer = max(middle_scores, key=middle_scores.get)

    half = window // 2
    candidates = list(range(max(0, best_layer - half),
                            min(num_layers, best_layer + half + 1)))
    candidates = [l for l in candidates if l in layer_scores]

    return {
        "best_middle_layer": best_layer,
        "best_layer_score": layer_scores[best_layer],
        "middle_range": [bottom, top],
        "window": window,
        "candidate_layers": candidates,
        "candidate_scores": {l: layer_scores[l] for l in candidates},
    }


def main():
    parser = argparse.ArgumentParser(description="ProSafePrune layer analyzer")
    parser.add_argument("--window", type=int, default=5,
                        help="Candidate layer window size")
    parser.add_argument("--aggregation", default="mean", choices=["mean", "median"])
    args = parser.parse_args()

    print_config()

    act_dir = get_activation_dir()
    if not os.path.exists(act_dir):
        print(f"Error: activation cache not found: {act_dir}")
        print("Run python -m prosafeprune.collect_activations first.")
        raise FileNotFoundError(act_dir)

    print(f"\nActivation cache: {act_dir}")

    validate_asset(act_dir, "activation_cache", {"pooling": POOLING})
    out_dir = get_layer_analysis_dir()
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n[1/4] Loading activation cache...")
    activations = load_activation_cache(act_dir)
    print(f"  Loaded {len(activations)} modules")

    print(f"\n[2/4] Computing module silhouette scores...")
    module_scores = compute_module_silhouettes(activations)
    print(f"  Computed {len(module_scores)} module scores")

    print(f"\n[3/4] Aggregating to layers (method={args.aggregation})...")
    layer_scores = aggregate_to_layers(module_scores, method=args.aggregation)

    if not layer_scores:
        raise RuntimeError("No layer scores were produced from the activation assets")
    num_layers = max(layer_scores.keys()) + 1
    print(f"  Num layers: {num_layers}")

    print(f"\n[4/4] Finding candidate layers (window={args.window})...")
    candidates = find_candidate_layers(layer_scores, num_layers, window=args.window)

    print(f"\n{'='*70}")
    print(f"Layer Selection Results")
    print(f"{'='*70}")
    print(f"  Best middle layer: {candidates['best_middle_layer']} "
          f"(score={candidates['best_layer_score']:.4f})")
    print(f"  Candidate layers:  {candidates['candidate_layers']}")
    print(f"  Window: {args.window}, Middle range: {candidates['middle_range']}")
    print(f"{'='*70}")

    module_csv = os.path.join(out_dir, "module_silhouette_scores.csv")
    with open(module_csv, 'w') as f:
        f.write("module_name,layer_id,silhouette_score\n")
        for name, score in sorted(module_scores.items()):
            lid = parse_layer_id(name)
            lid = -1 if lid is None else lid
            f.write(f"{name},{lid},{score:.6f}\n")

    layer_csv = os.path.join(out_dir, "layer_scores.csv")
    with open(layer_csv, 'w') as f:
        f.write("layer_id,aggregated_score\n")
        for lid in sorted(layer_scores.keys()):
            f.write(f"{lid},{layer_scores[lid]:.6f}\n")

    candidates_json = os.path.join(out_dir, "candidate_layers.json")
    with open(candidates_json, 'w') as f:
        json.dump(candidates, f, indent=2)

    manifest = {
        "asset_type": "layer_analysis",
        "activation_dir": act_dir,
        "pooling": POOLING,
        "aggregation_method": args.aggregation,
        "num_layers": num_layers,
        "total_modules_analyzed": len(module_scores),
        "results": candidates,
    }
    save_manifest(out_dir, manifest)

    print("\nLayer analysis complete.")
    print(f"   Results: {out_dir}")
    print(f"   Files: module_silhouette_scores.csv, layer_scores.csv, candidate_layers.json")


if __name__ == "__main__":
    main()
