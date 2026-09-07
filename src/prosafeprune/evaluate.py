"""Evaluate Hugging Face causal language models with WildGuard.

The CLI provides separate ``generate``, ``score``, and ``report`` stages, plus
a ``full`` command that runs the complete evaluation pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch
from tqdm import tqdm

from .config import (
    MODEL_ID,
    MODEL_PATH,
    WILDGUARD_PATH,
    DEVICE_MAP,
    DTYPE,
    CHAT_TEMPLATE_KWARGS,
    EVAL_DATASETS,
    COMPLIANCE_DATASETS,
    SAFETY_DATASETS,
    WG_FORMAT,
    DEFAULT_MAX_NEW_TOKENS_GEN,
    DEFAULT_MAX_NEW_TOKENS_WG,
    DEFAULT_EVAL_SAMPLE_COUNT,
    compact_module_spec,
    get_eval_dir,
    infer_evaluation_label,
    TRUST_REMOTE_CODE,
    LOCAL_FILES_ONLY,
    save_manifest,
    print_config,
)
from .runtime import RuntimeOptions, input_device, load_causal_lm, load_tokenizer, move_inputs as move_model_inputs, render_chat_batch


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def evaluation_model_identity(model_path: Optional[str]) -> Dict[str, Any]:
    """Copy stable experiment identity into the final evaluation manifest."""
    if not model_path:
        return {"model_kind": "unknown"}
    candidate = str(Path(model_path).expanduser().resolve())
    configured = str(Path(MODEL_PATH).expanduser().resolve())
    if candidate == configured:
        return {"model_kind": "baseline"}

    directory = Path(candidate)
    if not directory.is_dir():
        directory = directory.parent
    for parent in (directory, *directory.parents):
        manifest_path = parent / "manifest.json"
        if manifest_path.is_file():
            manifest = load_json(str(manifest_path))
            if manifest.get("asset_type") == "pruned_model":
                module_spec = manifest.get("module_spec")
                if not module_spec and manifest.get("target_modules"):
                    module_spec = compact_module_spec(manifest["target_modules"])
                return {
                    "model_kind": "pruned",
                    "target_layers": manifest.get("target_layers"),
                    "rank": manifest.get("rank"),
                    "lambda": manifest.get("lambda"),
                    "module_spec": module_spec,
                    "target_modules": manifest.get("target_modules"),
                    "source_pruning_manifest": str(manifest_path),
                }
    return {"model_kind": "external"}


def sample_dataset(path: str, n: int, seed: int = 42) -> list:
    data = load_json(path)
    if not isinstance(data, list):
        raise TypeError(f"Dataset must be a JSON list: {path}")
    if n <= 0:
        raise ValueError("--sample-count must be positive")
    if n >= len(data):
        return data
    return random.Random(seed).sample(data, n)


def extract_prompt(item: Any) -> str:
    if isinstance(item, str):
        prompt = item
    elif isinstance(item, Mapping):
        prompt = (
            item.get("prompt")
            or item.get("question")
            or item.get("input")
            or item.get("instruction")
            or ""
        )
    else:
        raise TypeError(f"Unsupported dataset item: {type(item).__name__}")

    prompt = str(prompt).strip()
    if not prompt:
        raise ValueError(f"Empty prompt in dataset item: {item}")
    return prompt


def release_model(model: Optional[torch.nn.Module]) -> None:
    if model is not None:
        del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def format_wildguard_prompt(prompt: str, response: str) -> str:
    return WG_FORMAT.format(prompt=prompt, response=response)


def parse_wildguard_output(text: str) -> Dict[str, bool]:
    result = {
        "Harmful request": False,
        "Response refusal": False,
        "Harmful response": False,
    }

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if ":" not in line:
            continue

        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip().lower()

        if key in result:
            result[key] = value.startswith("yes")

    return result


# ============================================================
# 1. Target model generation
# ============================================================

def generate_responses(
    model_path: str,
    output_dir: str,
    datasets: Optional[Dict[str, str]] = None,
    sample_count: int = DEFAULT_EVAL_SAMPLE_COUNT,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS_GEN,
    seed: int = 42,
    force: bool = False,
    batch_size: int = 1,
    runtime_options: Optional[RuntimeOptions] = None,
) -> Dict[str, list]:
    if datasets is None:
        datasets = EVAL_DATASETS
    runtime_options = runtime_options or RuntimeOptions(
        DEVICE_MAP, DTYPE, TRUST_REMOTE_CODE, LOCAL_FILES_ONLY
    )

    os.makedirs(output_dir, exist_ok=True)
    cache_path = os.path.join(output_dir, "generations.json")
    cache_meta_path = os.path.join(output_dir, "generation_manifest.json")
    generation_config = {
        "decoder": "greedy",
        "model_path": os.path.abspath(os.path.expanduser(model_path)),
        "sample_count": sample_count,
        "max_new_tokens": max_new_tokens,
        "seed": seed,
        "chat_template_kwargs": CHAT_TEMPLATE_KWARGS,
        "device_map": runtime_options.device_map,
        "dtype": runtime_options.dtype,
        "trust_remote_code": runtime_options.trust_remote_code,
    }

    if os.path.exists(cache_path) and not force:
        if not os.path.exists(cache_meta_path) or load_json(cache_meta_path) != generation_config:
            raise FileExistsError(
                f"Generation cache inputs do not match: {output_dir}. "
                "Choose another --label or use --force."
            )
        print(f"[Cache] Loading generations: {cache_path}")
        return load_json(cache_path)

    print(f"\n[Generate] Loading model: {model_path}")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    tokenizer = load_tokenizer(model_path, runtime_options)
    model = load_causal_lm(model_path, runtime_options)
    model.eval()

    device = input_device(model)
    print("  model class:", type(model))
    print("  input device:", device)
    print("  device map:", getattr(model, "hf_device_map", None))

    generations: Dict[str, list] = {}

    try:
        for dataset_name, dataset_path in datasets.items():
            samples = sample_dataset(dataset_path, sample_count, seed)
            pairs = []

            prompts = [extract_prompt(item) for item in samples]
            for start in tqdm(range(0, len(prompts), batch_size), desc=f"Generate/{dataset_name}"):
                prompt_batch = prompts[start:start + batch_size]
                rendered = render_chat_batch(tokenizer, prompt_batch, CHAT_TEMPLATE_KWARGS)
                encoded = tokenizer(
                    rendered,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                )
                encoded = move_model_inputs(encoded, model)
                prompt_length = encoded["input_ids"].shape[1]

                with torch.inference_mode():
                    output_ids = model.generate(
                        **encoded,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        use_cache=True,
                    )

                responses = tokenizer.batch_decode(output_ids[:, prompt_length:], skip_special_tokens=True)
                pairs.extend(
                    {"prompt": prompt, "response": response.strip()}
                    for prompt, response in zip(prompt_batch, responses)
                )

            generations[dataset_name] = pairs

    finally:
        release_model(model)

    save_json(generations, cache_path)
    save_json(generation_config, cache_meta_path)
    print(f"Saved: {cache_path}")
    return generations


# ============================================================
# 2. WildGuard scoring
# ============================================================

def score_with_wildguard(
    generations: Dict[str, list],
    output_dir: str,
    wildguard_path: str = WILDGUARD_PATH,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS_WG,
    force: bool = False,
    runtime_options: Optional[RuntimeOptions] = None,
) -> Dict[str, dict]:
    os.makedirs(output_dir, exist_ok=True)
    runtime_options = runtime_options or RuntimeOptions(
        DEVICE_MAP, "auto", TRUST_REMOTE_CODE, LOCAL_FILES_ONLY
    )
    summary_path = os.path.join(output_dir, "wildguard_results.json")
    details_path = os.path.join(output_dir, "wildguard_details.json")
    cache_meta_path = os.path.join(output_dir, "scoring_manifest.json")
    scoring_config = {
        "wildguard_path": os.path.abspath(os.path.expanduser(wildguard_path)),
        "max_new_tokens": max_new_tokens,
        "device_map": runtime_options.device_map,
        "dtype": runtime_options.dtype,
        "trust_remote_code": runtime_options.trust_remote_code,
        "generations_sha256": hashlib.sha256(
            json.dumps(generations, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
    }

    if os.path.exists(summary_path) and not force:
        if not os.path.exists(cache_meta_path) or load_json(cache_meta_path) != scoring_config:
            raise FileExistsError(
                f"Scoring cache inputs do not match: {output_dir}. Use --force to rescore."
            )
        print(f"[Cache] Loading WildGuard results: {summary_path}")
        return load_json(summary_path)

    print(f"\n[WildGuard] Loading model: {wildguard_path}")

    tokenizer = load_tokenizer(wildguard_path, runtime_options)
    model = load_causal_lm(wildguard_path, runtime_options)
    model.eval()

    device = input_device(model)
    print("  model class:", type(model))
    print("  input device:", device)
    print("  device map:", getattr(model, "hf_device_map", None))

    results: Dict[str, dict] = {}
    all_details: Dict[str, list] = {}

    total_pairs = sum(len(pairs) for pairs in generations.values())
    progress = tqdm(total=total_pairs, desc="WildGuard")

    try:
        for dataset_name, pairs in generations.items():
            counts = {
                "Harmful request": 0,
                "Response refusal": 0,
                "Harmful response": 0,
            }
            details = []

            for pair in pairs:
                classifier_prompt = format_wildguard_prompt(
                    prompt=pair["prompt"],
                    response=pair["response"],
                )

                encoded = tokenizer(
                    classifier_prompt,
                    return_tensors="pt",
                    add_special_tokens=False,
                )
                encoded = move_model_inputs(encoded, model)
                prompt_length = encoded["input_ids"].shape[1]

                with torch.inference_mode():
                    output_ids = model.generate(
                        **encoded,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        use_cache=True,
                    )

                raw_output = tokenizer.decode(
                    output_ids[0, prompt_length:],
                    skip_special_tokens=True,
                ).strip()

                parsed = parse_wildguard_output(raw_output)

                for key in counts:
                    if parsed[key]:
                        counts[key] += 1

                details.append({
                    "prompt": pair["prompt"],
                    "response": pair["response"],
                    "wildguard_raw": raw_output,
                    "wildguard": parsed,
                })
                progress.update(1)

            total = len(pairs)
            results[dataset_name] = {
                "total": total,
                "counts": counts,
                "rates": {
                    key: (value / total * 100.0) if total else 0.0
                    for key, value in counts.items()
                },
            }
            all_details[dataset_name] = details

    finally:
        progress.close()
        release_model(model)

    save_json(results, summary_path)
    save_json(all_details, details_path)
    save_json(scoring_config, cache_meta_path)
    print(f"Saved: {summary_path}")
    print(f"Saved: {details_path}")
    return results


# ============================================================
# 3. Metrics and report
# ============================================================

def compute_metrics(
    wildguard_results: Dict[str, dict],
) -> Dict[str, Any]:
    compliance = {}
    safety = {}

    for dataset_name in COMPLIANCE_DATASETS:
        if dataset_name in wildguard_results:
            refusal_rate = wildguard_results[dataset_name]["rates"][
                "Response refusal"
            ]
            compliance[dataset_name] = 100.0 - refusal_rate

    for dataset_name in SAFETY_DATASETS:
        if dataset_name in wildguard_results:
            harmful_rate = wildguard_results[dataset_name]["rates"][
                "Harmful response"
            ]
            safety[dataset_name] = 100.0 - harmful_rate

    compliance_mean = (
        sum(compliance.values()) / len(compliance)
        if compliance else 0.0
    )
    safety_mean = (
        sum(safety.values()) / len(safety)
        if safety else 0.0
    )

    return {
        "compliance": {
            "per_dataset": compliance,
            "mean": compliance_mean,
        },
        "safety": {
            "per_dataset": safety,
            "mean": safety_mean,
        },
        "tradeoff_score": (compliance_mean + safety_mean) / 2.0,
    }


def format_report(metrics: Dict[str, Any], label: str) -> str:
    lines = [
        "=" * 60,
        f"Evaluation result: {label}",
        "=" * 60,
        "",
        "Compliance datasets:",
    ]

    for name, value in metrics["compliance"]["per_dataset"].items():
        lines.append(f"  {name:<24} {value:>8.2f}%")

    lines.extend([
        f"  {'Mean':<24} {metrics['compliance']['mean']:>8.2f}%",
        "",
        "Safety datasets:",
    ])

    for name, value in metrics["safety"]["per_dataset"].items():
        lines.append(f"  {name:<24} {value:>8.2f}%")

    lines.extend([
        f"  {'Mean':<24} {metrics['safety']['mean']:>8.2f}%",
        "",
        f"Tradeoff score: {metrics['tradeoff_score']:.2f}%",
        "=" * 60,
        "",
    ])

    return "\n".join(lines)


def save_report(
    metrics: Dict[str, Any],
    output_dir: str,
    label: str,
    model_path: Optional[str] = None,
    sample_count: Optional[int] = None,
    max_new_tokens: Optional[int] = None,
    seed: Optional[int] = None,
    runtime_options: Optional[RuntimeOptions] = None,
    decoder: Optional[Dict[str, Any]] = None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    generation_manifest_path = os.path.join(output_dir, "generation_manifest.json")
    if model_path is None and os.path.isfile(generation_manifest_path):
        generation_manifest = load_json(generation_manifest_path)
        model_path = generation_manifest.get("model_path")
        sample_count = generation_manifest.get("sample_count", sample_count)
        max_new_tokens = generation_manifest.get("max_new_tokens", max_new_tokens)
        seed = generation_manifest.get("seed", seed)

    summary_path = os.path.join(output_dir, "eval_summary.json")
    report_path = os.path.join(output_dir, "eval_report.txt")

    save_json({label: metrics}, summary_path)

    report_text = format_report(metrics, label)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    manifest = {
        "asset_type": "eval_results",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_id": MODEL_ID,
        "label": label,
        "model_path": model_path,
        "sample_count": sample_count,
        "max_new_tokens": max_new_tokens,
        "seed": seed,
        "device_map": runtime_options.device_map if runtime_options else DEVICE_MAP,
        "dtype": runtime_options.dtype if runtime_options else DTYPE,
        "trust_remote_code": (
            runtime_options.trust_remote_code if runtime_options else TRUST_REMOTE_CODE
        ),
        "chat_template_kwargs": CHAT_TEMPLATE_KWARGS,
        "decoder": decoder or {"name": "greedy"},
        **evaluation_model_identity(model_path),
    }
    save_manifest(output_dir, manifest)

    print(f"Saved: {summary_path}")
    print(f"Saved: {report_path}")
    print(report_text)


# ============================================================
# 4. Full evaluation
# ============================================================

def run_full_eval(
    model_path: str,
    output_dir: str,
    label: str,
    sample_count: int = DEFAULT_EVAL_SAMPLE_COUNT,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS_GEN,
    seed: int = 42,
    skip_generation: bool = False,
    skip_scoring: bool = False,
    force_generation: bool = False,
    force_scoring: bool = False,
    batch_size: int = 1,
    runtime_options: Optional[RuntimeOptions] = None,
) -> Dict[str, Any]:
    generations_path = os.path.join(output_dir, "generations.json")
    wildguard_path = os.path.join(
        output_dir,
        "wildguard_results.json",
    )

    if skip_generation:
        if not os.path.exists(generations_path):
            raise FileNotFoundError(generations_path)
        generations = load_json(generations_path)
    else:
        generations = generate_responses(
            model_path=model_path,
            output_dir=output_dir,
            sample_count=sample_count,
            max_new_tokens=max_new_tokens,
            seed=seed,
            force=force_generation,
            batch_size=batch_size,
            runtime_options=runtime_options,
        )

    if skip_scoring:
        if not os.path.exists(wildguard_path):
            raise FileNotFoundError(wildguard_path)
        wildguard_results = load_json(wildguard_path)
    else:
        wildguard_results = score_with_wildguard(
            generations=generations,
            output_dir=output_dir,
            force=force_scoring,
            runtime_options=runtime_options,
        )

    metrics = compute_metrics(wildguard_results)

    save_report(
        metrics=metrics,
        output_dir=output_dir,
        label=label,
        model_path=model_path,
        sample_count=sample_count,
        max_new_tokens=max_new_tokens,
        seed=seed,
        runtime_options=runtime_options,
        decoder={"name": "greedy"},
    )
    return metrics


# ============================================================
# CLI
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a Hugging Face causal language model with ProSafePrune tools"
    )
    sub = parser.add_subparsers(dest="command")

    p_generate = sub.add_parser("generate")
    p_generate.add_argument("--model-path", required=True)
    p_generate.add_argument("--label", default=None)
    p_generate.add_argument(
        "--sample-count",
        type=int,
        default=DEFAULT_EVAL_SAMPLE_COUNT,
    )
    p_generate.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS_GEN,
    )
    p_generate.add_argument("--seed", type=int, default=42)
    p_generate.add_argument("--force", action="store_true")
    p_generate.add_argument("--batch-size", type=int, default=1)

    p_score = sub.add_parser("score")
    p_score.add_argument("--generations-file", required=True)
    p_score.add_argument("--force", action="store_true")

    p_report = sub.add_parser("report")
    p_report.add_argument("--wildguard-results", required=True)
    p_report.add_argument("--label", default=None)

    p_full = sub.add_parser("full")
    p_full.add_argument("--model-path", required=True)
    p_full.add_argument("--label", default=None)
    p_full.add_argument(
        "--sample-count",
        type=int,
        default=DEFAULT_EVAL_SAMPLE_COUNT,
    )
    p_full.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS_GEN,
    )
    p_full.add_argument("--seed", type=int, default=42)
    p_full.add_argument("--skip-generation", action="store_true")
    p_full.add_argument("--skip-scoring", action="store_true")
    p_full.add_argument("--force-generation", action="store_true")
    p_full.add_argument("--force-scoring", action="store_true")
    p_full.add_argument("--batch-size", type=int, default=1)

    for subparser in (p_generate, p_score, p_full):
        subparser.add_argument("--device-map", default=DEVICE_MAP)
        subparser.add_argument("--dtype", default=DTYPE, choices=["auto", "float32", "float16", "bfloat16"])
        subparser.add_argument(
            "--trust-remote-code",
            action=argparse.BooleanOptionalAction,
            default=TRUST_REMOTE_CODE,
            help="Allow model repositories to execute their custom Python loading code",
        )
        subparser.add_argument("--allow-download", action="store_true")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    print_config()
    if args.command in {"generate", "full"}:
        args.label = infer_evaluation_label(args.model_path, args.label)
        args.output_dir = get_eval_dir(args.label)
    elif args.command == "score":
        args.output_dir = os.path.dirname(os.path.abspath(args.generations_file))
    else:
        args.output_dir = os.path.dirname(os.path.abspath(args.wildguard_results))
        args.label = args.label or os.path.basename(args.output_dir)
    runtime_options = None
    if hasattr(args, "device_map"):
        runtime_options = RuntimeOptions(
            args.device_map, args.dtype, args.trust_remote_code,
            not args.allow_download,
        )

    if args.command == "generate":
        generate_responses(
            model_path=args.model_path,
            output_dir=args.output_dir,
            sample_count=args.sample_count,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
            force=args.force,
            batch_size=args.batch_size,
            runtime_options=runtime_options,
        )

    elif args.command == "score":
        generations = load_json(args.generations_file)
        score_with_wildguard(
            generations=generations,
            output_dir=args.output_dir,
            force=args.force,
            runtime_options=runtime_options,
        )

    elif args.command == "report":
        wildguard_results = load_json(args.wildguard_results)
        metrics = compute_metrics(wildguard_results)
        save_report(
            metrics=metrics,
            output_dir=args.output_dir,
            label=args.label,
        )

    elif args.command == "full":
        run_full_eval(
            model_path=args.model_path,
            output_dir=args.output_dir,
            label=args.label,
            sample_count=args.sample_count,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
            skip_generation=args.skip_generation,
            skip_scoring=args.skip_scoring,
            force_generation=args.force_generation,
            force_scoring=args.force_scoring,
            batch_size=args.batch_size,
            runtime_options=runtime_options,
        )


if __name__ == "__main__":
    main()
