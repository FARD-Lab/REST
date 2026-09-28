from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from hf_resolver import resolve_inner_adapter
from huggingface_hub import snapshot_download
from load_from_repo import STYLE_SPECS

from rest import REPO_ROOT

STYLES = ("sequential_light", "sequential_scaled")

TRAIN_DATASET = "RecursiveMAS/Sequential-Math"
TRAIN_SPLIT = "train"

BASE_MODELS = {
    "sequential_light": {
        "planner": "Qwen/Qwen3-1.7B",
        "critic": "meta-llama/Llama-3.2-1B-Instruct",
        "solver": "Qwen/Qwen2.5-Math-1.5B-Instruct",
    },
    "sequential_scaled": {
        "planner": "google/gemma-3-4b-it",
        "critic": "meta-llama/Llama-3.2-3B-Instruct",
        "solver": "Qwen/Qwen3.5-4B",
    },
}

INNER_ADAPTER_TASK = "math"
INNER_ADAPTER_TYPE_FALLBACK = "res_adapter"
OUTER_ADAPTER_TYPE = "outer_ln_res_adapter"

ADAPTER_ROOT = REPO_ROOT / "adapters"
_INNER_ADAPTER_FILES = ["innerlink_config.json", "adapter_config.json", "adapter*.pt"]


def inner_adapter_repo(style: str, role: str) -> str:
    return str(STYLE_SPECS[style]["repos"][role])


def _atomic_copy(src: Path, dst: Path) -> None:
    fd, tmp = tempfile.mkstemp(dir=dst.parent, prefix=f".{dst.name}.")
    os.close(fd)
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def inner_adapter_path(style: str, role: str) -> str:
    repo_dir = Path(snapshot_download(inner_adapter_repo(style, role), allow_patterns=_INNER_ADAPTER_FILES))
    weights = resolve_inner_adapter(repo_dir, INNER_ADAPTER_TASK)
    config = repo_dir / "adapter_config.json"
    if not config.is_file():
        raise FileNotFoundError(f"{inner_adapter_repo(style, role)} has no adapter_config.json")
    out_dir = ADAPTER_ROOT / style / role
    out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_copy(weights, out_dir / "adapter.pt")
    _atomic_copy(config, out_dir / "adapter_config.json")
    return str(out_dir)
