from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import List, Optional, Tuple

import torch
from common import write_outerlink_manifest
from model import CrossModelAdapter

from rest.axioms import AxiomTerm
from rest.recipe import BASE_MODELS, INNER_ADAPTER_TASK, TRAIN_DATASET, inner_adapter_repo


def cleanup_intermediate_checkpoints(save_dir: str) -> None:
    if not os.path.isdir(save_dir):
        return
    for entry in os.listdir(save_dir):
        if entry.startswith("checkpoint-"):
            shutil.rmtree(os.path.join(save_dir, entry), ignore_errors=True)


def save_selfloop_checkpoint(
    save_dir: str,
    step: Optional[int],
    outer_self: CrossModelAdapter,
    args: argparse.Namespace,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    global_step: Optional[int] = None,
    axiom_terms: Optional[List[Tuple[str, AxiomTerm, float]]] = None,
) -> None:
    output_dir = os.path.join(save_dir, f"checkpoint-{step}") if step is not None else save_dir
    os.makedirs(output_dir, exist_ok=True)

    torch.save(outer_self.state_dict(), os.path.join(output_dir, "outer_self.pt"))

    if axiom_terms:
        per_axiom_states = {name: term.state_dict() for name, term, _ in axiom_terms}
        if any(per_axiom_states.values()):
            torch.save(
                {"axiom_names": [name for name, _, _ in axiom_terms], "states": per_axiom_states},
                os.path.join(output_dir, "axiom_term.pt"),
            )

    if optimizer is not None and scheduler is not None and global_step is not None:
        torch.save(
            {
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "global_step": global_step,
            },
            os.path.join(output_dir, "training_state.pt"),
        )

    cfg = {
        "outer_self_type": outer_self.adapter_type,
        "outer_self_in_dim": outer_self.in_dim,
        "outer_self_out_dim": outer_self.out_dim,
        "style": args.style,
        "train_dataset": TRAIN_DATASET,
        "base_model": BASE_MODELS[args.style]["solver"],
        "inner_adapter": f"{inner_adapter_repo(args.style, 'solver')}:{INNER_ADAPTER_TASK}",
        "enable_thinking": args.enable_thinking,
        "num_recursive_rounds": args.num_recursive_rounds,
        "axiom": args.axiom,
        "axiom_weight": args.axiom_weight,
    }
    with open(os.path.join(output_dir, "outer_adapter_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)
    with open(os.path.join(output_dir, "train_args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)
    write_outerlink_manifest(
        output_dir,
        "solver_selfloop",
        [
            {
                "legacy_key": "outer_self",
                "filename": "outer_self.pt",
                "adapter_type": outer_self.adapter_type,
                "in_dim": outer_self.in_dim,
                "out_dim": outer_self.out_dim,
            },
        ],
    )


def load_resume_checkpoint(
    resume_from: str,
    outer_self: CrossModelAdapter,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    axiom_terms: Optional[List[Tuple[str, AxiomTerm, float]]] = None,
) -> int:
    outer_self.load_state_dict(torch.load(os.path.join(resume_from, "outer_self.pt"), map_location=device))

    if axiom_terms and any(term.state_dict() for _, term, _ in axiom_terms):
        axiom_state_path = os.path.join(resume_from, "axiom_term.pt")
        if not os.path.isfile(axiom_state_path):
            raise FileNotFoundError(
                f"--resume_from {resume_from} has no axiom_term.pt but this run's axiom(s) have "
                "trainable state that must be restored -- cannot resume correctly."
            )
        raw = torch.load(axiom_state_path, map_location=device)
        saved_names = raw["axiom_names"]
        current_names = [name for name, _, _ in axiom_terms]
        if saved_names != current_names:
            raise RuntimeError(
                f"--resume_from {resume_from}'s axiom composition {saved_names} does not "
                f"match the current run's --axiom {current_names}."
            )
        for name, term, _ in axiom_terms:
            term.load_state_dict(raw["states"][name])

    state_path = os.path.join(resume_from, "training_state.pt")
    if not os.path.isfile(state_path):
        raise FileNotFoundError(
            f"--resume_from {resume_from} has no training_state.pt (optimizer/scheduler/step) -- "
            "cannot resume correctly. Only checkpoints saved by this file's --save_steps support "
            "--resume_from."
        )
    state = torch.load(state_path, map_location=device)
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler.load_state_dict(state["scheduler_state_dict"])
    return int(state["global_step"])


def load_outer_self_adapter(
    checkpoint_dir: str,
    hidden_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> CrossModelAdapter:
    state_path = os.path.join(checkpoint_dir, "outer_self.pt")
    if not os.path.isfile(state_path):
        raise FileNotFoundError(f"Outer self-loop adapter weights not found: {state_path}")

    config_path = os.path.join(checkpoint_dir, "outer_adapter_config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Outer self-loop adapter config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    adapter = CrossModelAdapter(hidden_size, hidden_size, cfg["outer_self_type"])
    adapter.load_state_dict(torch.load(state_path, map_location="cpu"))
    adapter.to(device=device, dtype=dtype)
    adapter.eval()
    for param in adapter.parameters():
        param.requires_grad = False
    return adapter
