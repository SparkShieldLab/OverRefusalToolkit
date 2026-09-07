"""Collect pooled activations from Hugging Face causal language models.

The collector supports ``device_map="auto"``, routes inputs to the embedding
device, hooks the configured linear modules, and moves mean-pooled activations
to CPU immediately to limit accelerator memory usage.
"""

import os
import json
import argparse
import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

from .config import (
    MODEL_ID,
    MODEL_PATH,
    DEVICE_MAP,
    DTYPE,
    CHAT_TEMPLATE_KWARGS,
    TRUST_REMOTE_CODE,
    LOCAL_FILES_ONLY,
    POOLING,
    SAMPLE_LIMITS,
    TARGET_MODULES,
    SAFE_DATA_FILE,
    UNSAFE_DATA_FILE,
    PSEUDO_DATA_FILE,
    get_activation_dir,
    save_manifest,
    load_manifest,
    print_config,
)
from .runtime import RuntimeOptions, discover_linear_modules, load_causal_lm, load_tokenizer, move_inputs, render_chat_batch


def _extract_tensor(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"Unsupported hook output type: {type(output)}")


def collect_activations(
    model,
    tokenizer,
    dataset: List[Tuple[str, str]],
    layer_names: List[str],
    max_samples: int = 200,
    batch_size: int = 1,
) -> Dict[str, torch.Tensor]:
    model.eval()
    layer_activations = {name: [] for name in layer_names}
    hook_outputs: Dict[str, torch.Tensor] = {}
    batch_mask = None

    def make_hook(name):
        def hook_fn(module, module_input, output):
            tensor = _extract_tensor(output)
            mask = batch_mask.to(tensor.device)
            weights = mask.to(tensor.dtype).unsqueeze(-1)
            pooled = (tensor * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
            hook_outputs[name] = pooled.detach().float().cpu()
        return hook_fn

    handles = []
    module_map = dict(model.named_modules())

    for name in layer_names:
        module = module_map.get(name)
        if module is None:
            raise KeyError(f"Module not found: {name}")
        if not isinstance(module, nn.Linear):
            raise TypeError(f"Target is not nn.Linear: {name} -> {type(module)}")
        handles.append(module.register_forward_hook(make_hook(name)))

    processed = 0

    try:
        prompts = [p for p, _ in dataset if isinstance(p, str) and p.strip()][:max_samples]
        for start in tqdm(range(0, len(prompts), batch_size), desc="  Collecting activations"):
            prompt_text = render_chat_batch(tokenizer, prompts[start:start + batch_size], CHAT_TEMPLATE_KWARGS)
            encoded = tokenizer(
                prompt_text,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )
            encoded = move_inputs(encoded, model)

            attention_mask = encoded.get("attention_mask")
            if attention_mask is None or attention_mask.sum().item() == 0:
                continue
            batch_mask = attention_mask

            hook_outputs.clear()

            with torch.inference_mode():
                _ = model(
                    input_ids=encoded["input_ids"],
                    attention_mask=attention_mask,
                    use_cache=False,
                )

            for name in layer_names:
                output = hook_outputs.get(name)
                if output is None:
                    continue

                layer_activations[name].extend(row.contiguous() for row in output)

            processed += len(prompt_text)

    finally:
        for handle in handles:
            handle.remove()
        hook_outputs.clear()

    print(f"  Processed valid samples: {processed}")

    return {
        name: torch.stack(vecs, dim=0) if vecs else torch.empty(0)
        for name, vecs in layer_activations.items()
    }


def load_dataset(data_path: str, preferred_key: str = "prompt"):
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    result = []
    for item in data:
        if isinstance(item, str):
            prompt = item
        elif isinstance(item, dict):
            prompt = (
                item.get(preferred_key)
                or item.get("prompt")
                or item.get("question")
                or item.get("input")
                or ""
            )
        else:
            continue
        if isinstance(prompt, str) and prompt.strip():
            result.append((prompt, ""))

    return result


def save_category_activations(
    activations: Dict[str, torch.Tensor],
    save_dir: str,
) -> int:
    os.makedirs(save_dir, exist_ok=True)
    saved = 0

    for name, tensor in activations.items():
        if tensor.numel() == 0:
            continue
        filename = name.replace(".", "--") + ".pt"
        torch.save(tensor, os.path.join(save_dir, filename))
        saved += 1

    return saved


def main():
    parser = argparse.ArgumentParser(description="ProSafePrune activation collection")
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--device-map", default=DEVICE_MAP)
    parser.add_argument("--dtype", default=DTYPE, choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=TRUST_REMOTE_CODE,
        help="Allow model repositories to execute their custom Python loading code",
    )
    parser.add_argument("--allow-download", action="store_true", help="Allow files not present in the local HF cache")
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Override the number of samples collected for each category",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove an existing activation cache before collecting again",
    )
    args = parser.parse_args()

    print_config()
    start_time = time.time()

    sample_limits = SAMPLE_LIMITS.copy()
    if args.samples is not None:
        if args.samples <= 0:
            raise ValueError("--samples must be positive")
        sample_limits = {k: args.samples for k in sample_limits}

    activation_dir = get_activation_dir()

    safe_dir = os.path.join(activation_dir, "safe")
    unsafe_dir = os.path.join(activation_dir, "unsafe")
    pseudo_dir = os.path.join(activation_dir, "pseudo")
    manifest_path = os.path.join(activation_dir, "manifest.json")

    print(f"\nActivation cache dir: {activation_dir}")

    if os.path.exists(manifest_path) and not args.force:
        existing = load_manifest(activation_dir)
        expected = {
            "model_path": os.path.abspath(os.path.expanduser(args.model_path)),
            "pooling": POOLING,
            "sample_limits": sample_limits,
            "target_modules_filter": TARGET_MODULES,
            "chat_template_kwargs": CHAT_TEMPLATE_KWARGS,
            "dtype": args.dtype,
            "device_map": args.device_map,
            "trust_remote_code": args.trust_remote_code,
        }
        actual = {
            "model_path": os.path.abspath(os.path.expanduser(existing.get("model_path", ""))),
            "pooling": existing.get("pooling"),
            "sample_limits": existing.get("sample_limits"),
            "target_modules_filter": existing.get("target_modules_filter"),
            "chat_template_kwargs": existing.get("chat_template_kwargs", {}),
            "dtype": existing.get("dtype"),
            "device_map": existing.get("device_map"),
            "trust_remote_code": existing.get("trust_remote_code", False),
        }
        if actual != expected:
            raise FileExistsError(
                f"Activation assets already exist with different inputs: {activation_dir}. "
                "Use --force only after confirming the old assets can be replaced."
            )
        print("\nActivation cache already exists; collection skipped.")
        print(f"Use --force to regenerate: {activation_dir}")
        return activation_dir

    if args.force and os.path.exists(activation_dir):
        import shutil
        print(f"Removing existing cache: {activation_dir}")
        shutil.rmtree(activation_dir)

    for directory in [safe_dir, unsafe_dir, pseudo_dir]:
        os.makedirs(directory, exist_ok=True)

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    options = RuntimeOptions(
        args.device_map, args.dtype, args.trust_remote_code,
        not args.allow_download,
    )
    print(f"\n[1/3] Loading model: {args.model_path}")
    tokenizer = load_tokenizer(args.model_path, options)
    model = load_causal_lm(args.model_path, options)
    model.eval()

    print("  Model class:", type(model))
    print("  Embedding device:", model.get_input_embeddings().weight.device)
    print("  Device map:", getattr(model, "hf_device_map", None))

    module_map = dict(model.named_modules())
    layer_names = list(discover_linear_modules(model, TARGET_MODULES))

    print(f"  Found {len(layer_names)} target Linear modules")
    print("  First modules:")
    for name in layer_names[:10]:
        module = module_map[name]
        print(f"    {name}: weight={tuple(module.weight.shape)}, device={module.weight.device}")

    expected = model.config.num_hidden_layers * len(TARGET_MODULES)
    if len(layer_names) != expected:
        print(
            f"  Warning: expected about {expected} modules "
            f"({model.config.num_hidden_layers} layers × {len(TARGET_MODULES)} targets), "
            f"but found {len(layer_names)}."
        )

    print("\n[2/3] Loading datasets...")
    safe_data = load_dataset(SAFE_DATA_FILE, "prompt")
    unsafe_data = load_dataset(UNSAFE_DATA_FILE, "question")
    pseudo_data = load_dataset(PSEUDO_DATA_FILE, "prompt")

    print(f"  Safe: {len(safe_data)}")
    print(f"  Unsafe: {len(unsafe_data)}")
    print(f"  Pseudo: {len(pseudo_data)}")

    print(f"\n[3/3] Collecting activations (pooling={POOLING})...")

    tasks = [
        ("safe", safe_data, safe_dir, sample_limits["safe"]),
        ("unsafe", unsafe_data, unsafe_dir, sample_limits["unsafe"]),
        ("pseudo", pseudo_data, pseudo_dir, sample_limits["pseudo"]),
    ]

    saved_counts = {}
    processed_sample_counts = {}

    for label, dataset, save_dir, max_samples in tasks:
        print(f"\n  --- {label}: max_samples={max_samples} ---")

        activations = collect_activations(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            layer_names=layer_names,
            max_samples=max_samples,
            batch_size=args.batch_size,
        )

        saved = save_category_activations(activations, save_dir)
        saved_counts[label] = saved
        processed_sample_counts[label] = next(
            (int(tensor.shape[0]) for tensor in activations.values() if tensor.numel()), 0
        )
        print(f"  Saved {saved} module activation files to {save_dir}")
        del activations

    manifest = {
        "asset_type": "activation_cache",
        "model_id": MODEL_ID,
        "model_path": args.model_path,
        "pooling": POOLING,
        "sample_limits": sample_limits,
        "processed_sample_counts": processed_sample_counts,
        "target_modules_filter": TARGET_MODULES,
        "total_modules_hooked": len(layer_names),
        "saved_module_counts": saved_counts,
        "layer_names_sample": layer_names[:5] + ["..."] + layer_names[-3:],
        "dtype": args.dtype,
        "device_map": args.device_map,
        "trust_remote_code": args.trust_remote_code,
        "batch_size": args.batch_size,
        "chat_template_kwargs": CHAT_TEMPLATE_KWARGS,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "code_version": "prosafeprune_v1",
    }
    save_manifest(activation_dir, manifest)

    elapsed = time.time() - start_time
    print(f"\nActivation collection complete ({elapsed:.1f}s)")
    print(f"Cache: {activation_dir}")
    print(f"Manifest: {manifest_path}")
    return activation_dir


if __name__ == "__main__":
    main()
