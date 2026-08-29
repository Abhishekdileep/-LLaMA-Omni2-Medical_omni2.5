"""Safetensors helpers shared by the training stages.

Each stage trains one slice of the 2S checkpoint and freezes the rest, so they
all need the same two operations: read back a named slice of the weights
without materialising the 5 GB shards, and write a complete checkpoint whose
frozen tensors are byte-identical to the source.

Keeping saves complete matters -- `inference.py`, `run.sh` and `bleuscore.py`
load `Omni2Speech2SQwen2ForCausalLM`, so a stage output that is missing
`speech_generator.*` (as the pre-fix Stage I(a) script produced) cannot be run
end to end.
"""

import json
import os
import shutil
from typing import Dict, List

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SAFETENSORS_INDEX = "model.safetensors.index.json"
SAFETENSORS_SINGLE = "model.safetensors"

# Everything a checkpoint directory needs alongside its weights.
SIDECAR_FILES = (
    SAFETENSORS_INDEX, "config.json", "generation_config.json",
    "tokenizer_config.json", "vocab.json", "merges.txt",
    "added_tokens.json", "special_tokens_map.json",
)
SIDECAR_DIRS = ("tts_tokenizer",)


def shard_paths(model_path: str) -> Dict[str, List[str]]:
    """Map each safetensors shard under `model_path` to the keys it holds."""
    index_path = os.path.join(model_path, SAFETENSORS_INDEX)
    if os.path.isfile(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        shards: Dict[str, List[str]] = {}
        for key, shard in weight_map.items():
            shards.setdefault(shard, []).append(key)
        return shards

    single = os.path.join(model_path, SAFETENSORS_SINGLE)
    if os.path.isfile(single):
        with safe_open(single, framework="pt", device="cpu") as f:
            return {SAFETENSORS_SINGLE: list(f.keys())}

    raise FileNotFoundError(f"No safetensors weights found under {model_path}")


def read_prefixed_tensors(model_path: str, prefix: str, strip: bool = True) -> Dict[str, torch.Tensor]:
    """Read only the tensors whose key starts with `prefix`, one shard at a time.

    `safe_open` rather than `load_file` so a 5 GB shard is never materialised in
    full just to pull a few hundred small tensors out of it.
    """
    state: Dict[str, torch.Tensor] = {}
    for shard, keys in shard_paths(model_path).items():
        wanted = [k for k in keys if k.startswith(prefix)]
        if not wanted:
            continue
        with safe_open(os.path.join(model_path, shard), framework="pt", device="cpu") as f:
            for key in wanted:
                state[key[len(prefix):] if strip else key] = f.get_tensor(key)
    return state


def prefixed_state_dict(module, prefix: str = "") -> Dict[str, torch.Tensor]:
    """State dict on CPU, cloned.

    The clone matters: tied weights (MTTS ties `lm_head` to its embedding) share
    storage, and `save_file` refuses tensors that do.
    """
    return {
        f"{prefix}{k}": v.detach().to("cpu").clone().contiguous()
        for k, v in module.state_dict().items()
    }


def merge_and_save_checkpoint(source_model_path: str, output_dir: str,
                              overrides: Dict[str, torch.Tensor],
                              require_all: bool = True) -> None:
    """Copy `source_model_path`'s shards to `output_dir`, replacing `overrides`.

    Streams one shard at a time, so peak memory is a single shard (~5 GB) rather
    than the whole 16 GB checkpoint -- which is what lets a stage save a full 2S
    checkpoint without ever having held the frozen parts on the GPU.  Tensors
    not in `overrides` are written back byte-identical.
    """
    os.makedirs(output_dir, exist_ok=True)
    replaced = 0

    for shard, keys in shard_paths(source_model_path).items():
        with safe_open(os.path.join(source_model_path, shard), framework="pt", device="cpu") as f:
            tensors = {}
            for key in keys:
                if key in overrides:
                    tensors[key] = overrides[key]
                    replaced += 1
                else:
                    tensors[key] = f.get_tensor(key).contiguous()
        save_file(tensors, os.path.join(output_dir, shard), metadata={"format": "pt"})
        del tensors

    if require_all and replaced != len(overrides):
        unmatched = replaced, len(overrides)
        raise RuntimeError(
            f"Wrote {unmatched[0]} of {unmatched[1]} overridden tensors; the source "
            f"checkpoint layout at {source_model_path} does not match the live model."
        )

    for name in SIDECAR_FILES:
        src = os.path.join(source_model_path, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(output_dir, name))
    for name in SIDECAR_DIRS:
        src = os.path.join(source_model_path, name)
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(output_dir, name), dirs_exist_ok=True)
