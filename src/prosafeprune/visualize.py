"""Visualize dense rank and layer analysis outputs."""

import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import get_layer_analysis_dir, get_rank_analysis_dir, model_dir


def plot_dense_rank(input_file, output_file):
    with open(input_file, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty rank analysis: {input_file}")
    ranks = [int(row["rank"]) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for key, label in (("su_u", "Safe-Unsafe"), ("sp_u", "Safe-Pseudo"), ("up_u", "Unsafe-Pseudo")):
        axes[0].plot(ranks, [float(row[key]) for row in rows], label=label)
    for key, label in (("su_vh", "Safe-Unsafe"), ("sp_vh", "Safe-Pseudo"), ("up_vh", "Unsafe-Pseudo")):
        axes[1].plot(ranks, [float(row[key]) for row in rows], label=label)
    axes[0].set_title("Feature-space overlap")
    axes[1].set_title("Sample-space overlap")
    for axis in axes:
        axis.set_xlabel("Rank")
        axis.set_ylabel("Overlap")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_file, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_layers(scores_file, candidates_file, output_file):
    with open(scores_file, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with open(candidates_file, encoding="utf-8") as handle:
        candidates = json.load(handle).get("candidate_layers", [])
    layers = [int(row["layer_id"]) for row in rows]
    scores = [float(row["aggregated_score"]) for row in rows]
    colors = ["#d62728" if layer in candidates else "#4c78a8" for layer in layers]
    fig, axis = plt.subplots(figsize=(12, 5))
    axis.bar(layers, scores, color=colors)
    axis.set_xlabel("Layer")
    axis.set_ylabel("Silhouette score")
    axis.set_title("Layer separability")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_file, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize ProSafePrune analysis")
    args = parser.parse_args()
    rank_dir = get_rank_analysis_dir()
    layer_dir = get_layer_analysis_dir()
    output_dir = os.path.join(model_dir(), "figures")
    os.makedirs(output_dir, exist_ok=True)

    generated = []
    missing = []
    dense_file = os.path.join(rank_dir, "overlap_rank_dense.csv")
    if os.path.isfile(dense_file):
        rank_output = os.path.join(output_dir, "rank_overlap.png")
        plot_dense_rank(dense_file, rank_output)
        generated.append(rank_output)
    else:
        missing.append(dense_file)
    scores_file = os.path.join(layer_dir, "layer_scores.csv")
    candidates_file = os.path.join(layer_dir, "candidate_layers.json")
    if os.path.isfile(scores_file) and os.path.isfile(candidates_file):
        layer_output = os.path.join(output_dir, "layer_scores.png")
        plot_layers(scores_file, candidates_file, layer_output)
        generated.append(layer_output)
    else:
        missing.extend(path for path in (scores_file, candidates_file) if not os.path.isfile(path))

    if not generated:
        raise FileNotFoundError(
            "No analysis results are available for visualization. Missing: " + ", ".join(missing)
        )
    for path in missing:
        print(f"Warning: visualization input not found: {path}")
    print("Generated figures:")
    for path in generated:
        print(f"  {path}")


if __name__ == "__main__":
    main()
