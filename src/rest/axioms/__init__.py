from __future__ import annotations

from typing import Optional, Sequence

import torch

from .base import AxiomTerm, HandoffContext
from .causality import CausalityTerm
from .codi_kd import CodiKdTerm
from .minimality import MinimalityTerm
from .separability import SeparabilityTerm
from .simcot_step import SimCotStepTerm
from .stability import StabilityTerm

AXIOM_TERMS = {
    "causality": CausalityTerm,
    "minimality": MinimalityTerm,
    "stability": StabilityTerm,
    "separability": SeparabilityTerm,
    "codi_kd": CodiKdTerm,
    "simcot_step": SimCotStepTerm,
}


def build_axiom_term(
    name: Optional[str],
    weight: float,
    planner_hidden_size: Optional[int] = None,
    refiner_hidden_size: Optional[int] = None,
    solver_hidden_size: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    planner_model: Optional[torch.nn.Module] = None,
    refiner_model: Optional[torch.nn.Module] = None,
    solver_model: Optional[torch.nn.Module] = None,
    composed: bool = False,
    simcot_stages: Optional[Sequence[str]] = None,
) -> Optional[AxiomTerm]:
    if not name or name == "none":
        return None
    if name not in AXIOM_TERMS:
        raise ValueError(f"Unknown axiom {name!r}. Available: {sorted(AXIOM_TERMS)}")
    return AXIOM_TERMS[name](
        weight,
        planner_hidden_size=planner_hidden_size,
        refiner_hidden_size=refiner_hidden_size,
        solver_hidden_size=solver_hidden_size,
        device=device,
        dtype=dtype,
        planner_model=planner_model,
        refiner_model=refiner_model,
        solver_model=solver_model,
        composed=composed,
        simcot_stages=simcot_stages,
    )


__all__ = ["AxiomTerm", "HandoffContext", "build_axiom_term", "AXIOM_TERMS"]
