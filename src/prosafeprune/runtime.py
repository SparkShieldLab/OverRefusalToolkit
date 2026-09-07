"""Shared Hugging Face runtime helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


DTYPES = {
    "auto": "auto",
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@dataclass(frozen=True)
class RuntimeOptions:
    device_map: Any = "auto"
    dtype: Any = "auto"
    trust_remote_code: bool = False
    local_files_only: bool = True
    attn_implementation: Optional[str] = None


def parse_device_map(value: Any) -> Any:
    """Parse HF/Accelerate device placement from a CLI-friendly value."""
    if value is None or isinstance(value, Mapping):
        return value
    text = str(value).strip()
    if text.lower() in {"none", "null"}:
        return None
    if text in {"auto", "balanced", "balanced_low_0", "sequential"}:
        return text
    if text.startswith("{"):
        mapping = json.loads(text)
        if not isinstance(mapping, dict):
            raise ValueError("JSON device map must be an object")
        return mapping
    if text == "cuda":
        text = "cuda:0"
    if text == "cpu" or text.startswith("cuda:"):
        return {"": text}
    raise ValueError(
        "Unsupported device map. Use auto, balanced, balanced_low_0, sequential, "
        "cpu, cuda:N, none, or a JSON object."
    )


def parse_dtype(value: Any) -> Any:
    if isinstance(value, torch.dtype):
        return value
    key = str(value).lower()
    if key not in DTYPES:
        raise ValueError(f"Unsupported dtype: {value}. Choose from {sorted(DTYPES)}")
    dtype = DTYPES[key]
    if dtype in {torch.float16, torch.bfloat16} and not torch.cuda.is_available():
        return torch.float32
    return dtype


def load_tokenizer(model_path: str, options: RuntimeOptions):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=options.local_files_only,
        trust_remote_code=options.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_causal_lm(model_path: str, options: RuntimeOptions):
    kwargs = {
        "device_map": parse_device_map(options.device_map),
        "dtype": parse_dtype(options.dtype),
        "low_cpu_mem_usage": True,
        "local_files_only": options.local_files_only,
        "trust_remote_code": options.trust_remote_code,
    }
    if options.attn_implementation:
        kwargs["attn_implementation"] = options.attn_implementation
    try:
        return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    except TypeError as exc:
        if "dtype" not in str(exc):
            raise
        kwargs["torch_dtype"] = kwargs.pop("dtype")
        return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)


def input_device(model: nn.Module) -> torch.device:
    """The only device callers need to place tokenized inputs on."""
    return model.get_input_embeddings().weight.device


def move_inputs(encoded: Mapping[str, torch.Tensor], model: nn.Module) -> Dict[str, torch.Tensor]:
    device = input_device(model)
    return {key: value.to(device) for key, value in encoded.items()}


def render_chat_messages(tokenizer, messages, chat_template_kwargs=None) -> str:
    """Render arbitrary messages through a tokenizer's native chat template."""
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("The tokenizer has no chat_template; configure one before using this model")
    extra = dict(chat_template_kwargs or {})
    reserved = {"conversation", "tokenize", "add_generation_prompt"}.intersection(extra)
    if reserved:
        raise ValueError(f"Reserved chat template arguments cannot be overridden: {sorted(reserved)}")
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **extra,
    )


def render_chat_prompt(tokenizer, prompt: str, chat_template_kwargs=None) -> str:
    """Render one user prompt through a tokenizer's native chat template."""
    return render_chat_messages(
        tokenizer,
        [{"role": "user", "content": prompt}],
        chat_template_kwargs,
    )


def render_chat_batch(tokenizer, prompts: Sequence[str], chat_template_kwargs=None):
    return [render_chat_prompt(tokenizer, item, chat_template_kwargs) for item in prompts]


def discover_linear_modules(
    model: nn.Module,
    target_suffixes: Iterable[str],
    exclude: Iterable[str] = ("lm_head",),
) -> Dict[str, nn.Linear]:
    """Match module path components exactly, avoiding accidental substring matches."""
    targets = set(target_suffixes)
    excluded = set(exclude)
    result = {}
    for name, module in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if isinstance(module, nn.Linear) and leaf in targets and leaf not in excluded:
            result[name] = module
    return result
