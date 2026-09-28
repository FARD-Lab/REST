import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RMAS_ROOT = REPO_ROOT / "refs" / "recursive_mas"
RMAS_TRAIN_DIR = RMAS_ROOT / "train"
RMAS_OUTER_DIR = RMAS_TRAIN_DIR / "outer"
RMAS_INFERENCE_DIR = RMAS_ROOT / "inference"

if not RMAS_TRAIN_DIR.is_dir():
    raise RuntimeError(f"RecursiveMAS not found at {RMAS_ROOT}. Run ./setup.sh from the repository root.")

for _path in (RMAS_TRAIN_DIR, RMAS_OUTER_DIR, RMAS_INFERENCE_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
