from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import AxiomTerm, HandoffContext
from .pooling import AttentionPool


def compute_answer_entropy(logits: torch.Tensor, assistant_mask: torch.Tensor) -> Optional[torch.Tensor]:
    if not bool(assistant_mask.any()):
        return None
    with torch.no_grad():
        answer_logits = logits[0, assistant_mask, :].float()
        probs = torch.softmax(answer_logits, dim=-1)
        log_probs = torch.log_softmax(answer_logits, dim=-1)
        return -(probs * log_probs).sum(dim=-1).mean()


class _StageState:
    def __init__(self, hidden_size: int, device: torch.device, dtype: torch.dtype):
        self.pool = AttentionPool(hidden_size).to(device=device, dtype=dtype)
        self.probe = nn.Linear(hidden_size, 1).to(device=device, dtype=dtype)

    def parameters(self) -> List[torch.nn.Parameter]:
        return list(self.pool.parameters()) + list(self.probe.parameters())

    def state_dict(self) -> Dict[str, object]:
        return {"pool": self.pool.state_dict(), "probe": self.probe.state_dict()}

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self.pool.load_state_dict(state["pool"])
        self.probe.load_state_dict(state["probe"])


class StabilityTerm(AxiomTerm):
    def __init__(
        self,
        weight: float,
        planner_hidden_size: int,
        refiner_hidden_size: int,
        solver_hidden_size: int,
        device: torch.device,
        dtype: torch.dtype,
        **kwargs,
    ):
        super().__init__(weight)
        self._stages: Dict[int, _StageState] = {}
        for hidden_size in {planner_hidden_size, refiner_hidden_size, solver_hidden_size}:
            self._stages[hidden_size] = _StageState(hidden_size, device, dtype)

    def parameters(self) -> List[torch.nn.Parameter]:
        params: List[torch.nn.Parameter] = []
        for stage in self._stages.values():
            params += stage.parameters()
        return params

    def state_dict(self) -> Dict[str, object]:
        return {str(dim): stage.state_dict() for dim, stage in self._stages.items()}

    def load_state_dict(self, state: Dict[str, object]) -> None:
        for dim_str, stage_state in state.items():
            self._stages[int(dim_str)].load_state_dict(stage_state)

    def compute(self, ctx: HandoffContext) -> Optional[torch.Tensor]:
        if ctx.vector.size(0) == 0 or ctx.upstream_logits is None:
            return None
        stage = self._stages.get(ctx.vector.size(-1))
        if stage is None:
            return None

        target_entropy = compute_answer_entropy(ctx.upstream_logits, ctx.upstream_assistant_mask)
        if target_entropy is None:
            return None

        pooled = stage.pool(ctx.vector.to(stage.pool.score.weight.dtype))
        predicted_entropy = stage.probe(pooled.to(stage.probe.weight.dtype)).squeeze(-1).float()
        return F.mse_loss(predicted_entropy, target_entropy.detach().float())
