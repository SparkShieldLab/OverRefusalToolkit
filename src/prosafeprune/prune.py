"""
Memory-efficient ProSafePrune projection pruning for Hugging Face causal language models.

Core formulation:
    Π_s = U_s U_s^T
    Π_u = U_u U_u^T
    Π_p = U_p U_p^T
    Ω   = (I - Π_s) Π_u Π_p
    W'  = W - λ Ω W

The implementation evaluates the algebraically equivalent low-rank form without
materializing D x D matrices. Bases are moved to the device of each target
weight, parameters are updated in place on that device, and results are saved
to a new model directory without overwriting the source model.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Dict, List

import torch
import torch.nn as nn

from .config import (
    MODEL_ID,
    MODEL_PATH,
    DEVICE_MAP,
    DTYPE,
    TRUST_REMOTE_CODE,
    LOCAL_FILES_ONLY,
    POOLING,
    parse_layer_id,
    get_subspace_dir,
    get_pruned_model_dir,
    save_manifest,
    print_config,
    validate_asset,
)
from .runtime import RuntimeOptions, load_causal_lm, load_tokenizer


def parse_layers(layers_text: str) -> List[int]:
    """Parse one layer, an inclusive range, or a non-contiguous layer list."""
    layers_text = layers_text.strip()
    if not layers_text:
        raise ValueError("--layers cannot be empty")
    if "-" in layers_text:
        start, end = layers_text.split("-", 1)
        start_i, end_i = int(start), int(end)
        if start_i > end_i:
            raise ValueError(f"Invalid layer range: {layers_text}")
        layers = list(range(start_i, end_i + 1))
    elif "_" in layers_text or "," in layers_text:
        separator = "_" if "_" in layers_text else ","
        layers = [int(x.strip()) for x in layers_text.split(separator) if x.strip()]
    else:
        layers = [int(layers_text)]

    if not layers or any(layer < 0 for layer in layers):
        raise ValueError("--layers must contain non-negative layer numbers")
    return list(dict.fromkeys(layers))


MODULE_GROUPS = {
    "7m": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "att": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "mlp": ["gate_proj", "up_proj", "down_proj"],
    "q": ["q_proj"],
    "k": ["k_proj"],
    "v": ["v_proj"],
    "o": ["o_proj"],
    "gate": ["gate_proj"],
    "up": ["up_proj"],
    "down": ["down_proj"],
}


def parse_modules(module_spec: str) -> List[str]:
    """Expand a named module group or accept explicit comma-separated leaf names."""
    value = module_spec.strip().lower()
    if value in MODULE_GROUPS:
        return list(MODULE_GROUPS[value])
    modules = [item.strip() for item in module_spec.split(",") if item.strip()]
    if not modules:
        raise ValueError("--modules must be a known group or contain module names")
    return list(dict.fromkeys(modules))


def normalize_module_spec(module_spec: str, modules: List[str]) -> str:
    value = module_spec.strip().lower()
    if value == "attn":
        return "att"
    if value in MODULE_GROUPS:
        return value
    return "-".join(item.removesuffix("_proj") for item in modules)


def compute_delta_explicit(
    weight: torch.Tensor,
    basis_safe: torch.Tensor,
    basis_unsafe: torch.Tensor,
    basis_pseudo: torch.Tensor,
) -> torch.Tensor:
    """Compute the pruning update using the original projection formula.

        ΔW = (I - UsUs^T) UuUu^T UpUp^T W

    Shapes:
        W  : [D, in_features]
        Us : [D, rank]
        Uu : [D, rank]
        Up : [D, rank]
    """
    # Associative low-rank form of (I-UsUs^T) UuUu^T UpUp^T W.
    # Peak intermediates are O(D*r + r*in_features), not O(D^2).
    pseudo_coeff = basis_pseudo.transpose(0, 1) @ weight
    pseudo_projected_in_unsafe = (basis_unsafe.transpose(0, 1) @ basis_pseudo) @ pseudo_coeff
    unsafe_component = basis_unsafe @ pseudo_projected_in_unsafe
    safe_component = basis_safe @ (basis_safe.transpose(0, 1) @ unsafe_component)
    return unsafe_component - safe_component


def load_subspace_maps(subspace_dir: str):
    """Load bases on CPU before moving them to target parameter devices."""
    safe_map = torch.load(
        os.path.join(subspace_dir, "safe_basis.pt"),
        map_location="cpu",
        weights_only=True,
    )
    unsafe_map = torch.load(
        os.path.join(subspace_dir, "unsafe_basis.pt"),
        map_location="cpu",
        weights_only=True,
    )
    pseudo_map = torch.load(
        os.path.join(subspace_dir, "pseudo_basis.pt"),
        map_location="cpu",
        weights_only=True,
    )
    return safe_map, unsafe_map, pseudo_map


def find_target_module_names(
    model: nn.Module,
    safe_map: Dict[str, torch.Tensor],
    unsafe_map: Dict[str, torch.Tensor],
    pseudo_map: Dict[str, torch.Tensor],
    target_layers: List[int],
    target_keys: List[str],
) -> List[str]:
    """Find target linear modules with all three category-specific bases."""
    module_map = dict(model.named_modules())
    common_names = set(safe_map) & set(unsafe_map) & set(pseudo_map)

    targets = []
    for name in sorted(common_names):
        layer_id = parse_layer_id(name)
        if layer_id not in target_layers:
            continue
        if name.rsplit(".", 1)[-1] not in target_keys:
            continue

        module = module_map.get(name)
        if module is None:
            print(f"[Skip] module not found: {name}")
            continue
        if not isinstance(module, nn.Linear):
            print(f"[Skip] not nn.Linear: {name} -> {type(module)}")
            continue

        targets.append(name)

    return targets


@torch.no_grad()
def apply_explicit_pruning(
    model: nn.Module,
    safe_map: Dict[str, torch.Tensor],
    unsafe_map: Dict[str, torch.Tensor],
    pseudo_map: Dict[str, torch.Tensor],
    target_names: List[str],
    rank: int,
    prune_weight: float,
):
    """Apply the projection update to target linear weights.

        W <- W - λ (I-Πs)ΠuΠpW
    """
    module_map = dict(model.named_modules())
    records = []

    for name in target_names:
        module = module_map[name]
        weight = module.weight
        device = weight.device
        dtype = weight.dtype

        basis_safe = safe_map[name][:, :rank].to(
            device=device,
            dtype=dtype,
        )
        basis_unsafe = unsafe_map[name][:, :rank].to(
            device=device,
            dtype=dtype,
        )
        basis_pseudo = pseudo_map[name][:, :rank].to(
            device=device,
            dtype=dtype,
        )

        if not (
            basis_safe.shape[0]
            == basis_unsafe.shape[0]
            == basis_pseudo.shape[0]
            == weight.shape[0]
        ):
            print(
                f"[Skip] shape mismatch: {name}: "
                f"W={tuple(weight.shape)}, "
                f"Us={tuple(basis_safe.shape)}, "
                f"Uu={tuple(basis_unsafe.shape)}, "
                f"Up={tuple(basis_pseudo.shape)}"
            )
            continue

        if min(
            basis_safe.shape[1],
            basis_unsafe.shape[1],
            basis_pseudo.shape[1],
        ) < rank:
            print(f"[Skip] basis rank smaller than requested rank: {name}")
            continue

        delta_weight = compute_delta_explicit(
            weight=weight,
            basis_safe=basis_safe,
            basis_unsafe=basis_unsafe,
            basis_pseudo=basis_pseudo,
        )

        max_delta = delta_weight.detach().float().abs().max().item()
        mean_delta = delta_weight.detach().float().abs().mean().item()

        weight.sub_(delta_weight, alpha=prune_weight)

        print(
            f"[Pruned] {name}: "
            f"W={tuple(weight.shape)}, rank={rank}, λ={prune_weight}, "
            f"device={device}, "
            f"max|ΔW|={max_delta:.8f}, mean|ΔW|={mean_delta:.8f}"
        )

        records.append(
            {
                "name": name,
                "layer_id": parse_layer_id(name),
                "weight_shape": list(weight.shape),
                "basis_shape": list(basis_safe.shape),
                "rank": rank,
                "lambda": prune_weight,
                "device": str(device),
                "dtype": str(dtype),
                "max_abs_delta": max_delta,
                "mean_abs_delta": mean_delta,
            }
        )

        del basis_safe
        del basis_unsafe
        del basis_pseudo
        del delta_weight

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return records


def main():
    parser = argparse.ArgumentParser(
        description="Apply ProSafePrune projection pruning to a Hugging Face causal language model"
    )
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--device-map", default=DEVICE_MAP)
    parser.add_argument("--dtype", default=DTYPE, choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=TRUST_REMOTE_CODE,
        help="Allow model repositories to execute their custom Python loading code",
    )
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument(
        "--rank",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--layers",
        required=True,
        help="Single: 16; continuous: 15-17; non-contiguous: 14_16_18 or 14,16,18",
    )
    parser.add_argument(
        "--lam",
        type=float,
        required=True,
        help="Pruning strength λ",
    )
    parser.add_argument(
        "--modules",
        default="7m",
        help=("Group: 7m, att/attn, mlp, q, k, v, o, gate, up, down; "
              "or explicit comma-separated Linear module leaf names"),
    )
    args = parser.parse_args()

    print_config()

    if args.rank <= 0:
        raise ValueError("--rank must be positive")
    if args.lam < 0:
        raise ValueError("--lam must be non-negative")

    target_layers = parse_layers(args.layers)
    target_keys = parse_modules(args.modules)
    module_spec = normalize_module_spec(args.modules, target_keys)

    subspace_dir = get_subspace_dir()
    subspace_manifest = validate_asset(subspace_dir, "subspaces", {"pooling": POOLING})
    max_rank = int(subspace_manifest["max_rank"])
    if args.rank > max_rank:
        raise ValueError(f"Requested rank={args.rank} exceeds subspace max_rank={max_rank}")

    options = RuntimeOptions(
        args.device_map, args.dtype, args.trust_remote_code,
        not args.allow_download,
    )
    print(f"\n[1/4] Loading model: {args.model_path}")
    tokenizer = load_tokenizer(args.model_path, options)
    model = load_causal_lm(args.model_path, options)
    model.eval()

    print("  Model class:", type(model))
    print("  Embedding device:", model.get_input_embeddings().weight.device)
    print("  Device map:", getattr(model, "hf_device_map", None))

    print(f"\n[2/4] Loading subspaces from: {subspace_dir}")
    safe_map, unsafe_map, pseudo_map = load_subspace_maps(subspace_dir)

    print(
        f"  safe={len(safe_map)}, "
        f"unsafe={len(unsafe_map)}, "
        f"pseudo={len(pseudo_map)}"
    )

    target_names = find_target_module_names(
        model=model,
        safe_map=safe_map,
        unsafe_map=unsafe_map,
        pseudo_map=pseudo_map,
        target_layers=target_layers,
        target_keys=target_keys,
    )

    print(f"\n[3/4] Matched {len(target_names)} target modules")
    for name in target_names:
        module = dict(model.named_modules())[name]
        print(
            f"  {name}: weight={tuple(module.weight.shape)}, "
            f"device={module.weight.device}"
        )

    if not target_names:
        raise RuntimeError(
            "No modules matched. Check basis keys, --layers, and --modules."
        )

    print(
        f"\n[4/4] Applying explicit pruning: "
        f"layers={target_layers}, modules={target_keys}, "
        f"rank={args.rank}, λ={args.lam}"
    )

    records = apply_explicit_pruning(
        model=model,
        safe_map=safe_map,
        unsafe_map=unsafe_map,
        pseudo_map=pseudo_map,
        target_names=target_names,
        rank=args.rank,
        prune_weight=args.lam,
    )

    if not records:
        raise RuntimeError("No weights were modified.")

    output_dir = get_pruned_model_dir(
        rank=args.rank,
        layers=target_layers,
        lam=args.lam,
        modules=module_spec,
    )
    final_dir = os.path.join(output_dir, "model")
    os.makedirs(final_dir, exist_ok=True)

    print(f"\nSaving pruned model to: {final_dir}")
    model.save_pretrained(
        final_dir,
        safe_serialization=True,
    )
    tokenizer.save_pretrained(final_dir)

    save_manifest(
        output_dir,
        {
            "asset_type": "pruned_model",
            "model_id": MODEL_ID,
            "source_model_path": args.model_path,
            "subspace_dir": subspace_dir,
            "projection_implementation": "low_rank_equivalent",
            "formula": "(I-Pi_safe) @ Pi_unsafe @ Pi_pseudo @ W",
            "rank": args.rank,
            "max_rank": max_rank,
            "subspace_created_at": subspace_manifest.get("created_at"),
            "pooling": POOLING,
            "target_layers": target_layers,
            "target_modules": target_keys,
            "module_spec": module_spec,
            "lambda": args.lam,
            "device_map": args.device_map,
            "dtype": args.dtype,
            "trust_remote_code": args.trust_remote_code,
            "num_modules_pruned": len(records),
            "pruned_modules": records,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "code_version": "prosafeprune_v1",
        },
    )

    print("\nPruning complete.")
    print("Saved:", final_dir)


if __name__ == "__main__":
    main()
