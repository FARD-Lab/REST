from __future__ import annotations

import functools
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_CODE_DATASET_ALIASES = {
    "mbppplus",
    "mbpp+",
    "evalplus/mbppplus",
    "lcb",
    "livecodebench",
    "livecodebench_v6",
    "livecodebench/code_generation_lite",
    "livecodebench/code_generation_lite:release_v6",
}


def _patch_build_cli_for_style(original):
    @functools.wraps(original)
    def patched(*args, **kwargs):
        call_args = kwargs.get("args", args[0] if args else None)
        family = kwargs.get("family", args[1] if len(args) > 1 else None)
        module_obj, cli = original(*args, **kwargs)
        if family == "sequential" and getattr(call_args, "result_jsonl", "") and "--result_jsonl" not in cli:
            cli = cli + ["--result_jsonl", str(call_args.result_jsonl)]
        return module_obj, cli

    return patched


def _patch_task_for_inner_repo(original):
    @functools.wraps(original)
    def patched(dataset, *args, **kwargs):
        key = str(dataset or "").strip().lower()
        if key in _CODE_DATASET_ALIASES:
            return "code"
        return original(dataset, *args, **kwargs)

    return patched


def load_patched_run_module(run_py_path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, run_py_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load run.py as a module from {run_py_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    module.build_cli_for_style = _patch_build_cli_for_style(module.build_cli_for_style)
    module.task_for_inner_repo = _patch_task_for_inner_repo(module.task_for_inner_repo)
    return module
