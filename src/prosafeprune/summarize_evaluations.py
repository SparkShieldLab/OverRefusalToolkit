"""Build summary tables from ProSafePrune evaluation results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .config import COMPLIANCE_DATASETS, SAFETY_DATASETS, compact_layer_spec, model_dir


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def layer_text(value: Any) -> str:
    if isinstance(value, list):
        return compact_layer_spec(value)
    return "" if value is None else str(value)


def collect_rows(evaluations_dir: Path) -> List[Dict[str, Any]]:
    """Collect complete Greedy evaluation results and their model metadata."""
    if not evaluations_dir.is_dir():
        raise FileNotFoundError(f"Evaluation directory not found: {evaluations_dir}")

    rows: List[Dict[str, Any]] = []
    for summary_path in sorted(evaluations_dir.glob("greedy/*/eval_summary.json")):
        run_dir = summary_path.parent
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.is_file():
            print(f"Warning: skipped evaluation without manifest: {run_dir}")
            continue

        summary = load_json(summary_path)
        manifest = load_json(manifest_path)
        label = manifest.get("label") or next(iter(summary), run_dir.name)
        metrics = summary.get(label)
        if metrics is None and len(summary) == 1:
            metrics = next(iter(summary.values()))
        if not isinstance(metrics, dict):
            print(f"Warning: skipped invalid evaluation summary: {summary_path}")
            continue

        compliance = metrics.get("compliance", {})
        safety = metrics.get("safety", {})
        row: Dict[str, Any] = {
            "model_kind": manifest.get("model_kind", "unknown"),
            "label": label,
            "layers": layer_text(manifest.get("target_layers")),
            "rank": manifest.get("rank", ""),
            "lambda": manifest.get("lambda", ""),
            "modules": manifest.get("module_spec", ""),
            "compliance_mean": compliance.get("mean", ""),
            "safety_mean": safety.get("mean", ""),
            "tradeoff_score": metrics.get("tradeoff_score", ""),
            "created_at": manifest.get("created_at", ""),
            "model_path": manifest.get("model_path", ""),
            "evaluation_dir": str(run_dir),
        }
        for dataset in COMPLIANCE_DATASETS:
            row[dataset] = compliance.get("per_dataset", {}).get(dataset, "")
        for dataset in SAFETY_DATASETS:
            row[dataset] = safety.get("per_dataset", {}).get(dataset, "")
        rows.append(row)

    if not rows:
        raise FileNotFoundError(f"No complete evaluation results found in: {evaluations_dir}")
    return rows


def numeric_sort(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("inf")


def first_layer(value: Any) -> float:
    try:
        return float(str(value).replace("_", "-").split("-", 1)[0])
    except (TypeError, ValueError):
        return float("inf")


def descending_score(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_table_set(output_dir: Path, rows: List[Dict[str, Any]]) -> None:
    fields = [
        "label",
        *COMPLIANCE_DATASETS,
        "compliance_mean",
        *SAFETY_DATASETS,
        "safety_mean",
        "tradeoff_score",
    ]
    summary_rows = sorted(rows, key=lambda row: (row["model_kind"] != "baseline", row["label"]))
    rank_rows = sorted(rows, key=lambda row: (
        numeric_sort(row["rank"]),
        first_layer(row["layers"]),
        numeric_sort(row["lambda"]),
        str(row["modules"]),
        row["label"],
    ))
    layer_rows = sorted(rows, key=lambda row: (
        first_layer(row["layers"]),
        numeric_sort(row["rank"]),
        numeric_sort(row["lambda"]),
        str(row["modules"]),
        row["label"],
    ))
    leaderboard = sorted(rows, key=lambda row: descending_score(row["tradeoff_score"]), reverse=True)
    leaderboard = [{"position": index, **row} for index, row in enumerate(leaderboard, 1)]

    outputs = {
        "evaluation_summary.csv": (summary_rows, fields),
        "evaluation_by_rank.csv": (rank_rows, fields),
        "evaluation_by_layer.csv": (layer_rows, fields),
        "evaluation_leaderboard.csv": (leaderboard, ["position", *fields]),
    }
    for filename, (content, columns) in outputs.items():
        path = output_dir / filename
        write_csv(path, content, columns)
        print(f"Saved: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize ProSafePrune evaluations")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    model_root = Path(model_dir())
    evaluations_dir = model_root / "evaluations"
    output_dir = Path(args.output_dir) if args.output_dir else model_root / "evaluation_tables"
    rows = collect_rows(evaluations_dir)
    write_table_set(output_dir, rows)


if __name__ == "__main__":
    main()
