from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional

import torch
import torch.nn.functional as F

from .base import AxiomTerm, HandoffContext
from .pooling import AttentionPool

QUEUE_SIZE = 64
TEMPERATURE = 0.1


class _StageState:
    def __init__(self, hidden_size: int, device: torch.device, dtype: torch.dtype):
        self.pool = AttentionPool(hidden_size).to(device=device, dtype=dtype)
        self.queue: Deque[torch.Tensor] = deque(maxlen=QUEUE_SIZE)

    def parameters(self) -> List[torch.nn.Parameter]:
        return list(self.pool.parameters())

    def state_dict(self) -> Dict[str, object]:
        return {"pool": self.pool.state_dict()}

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self.pool.load_state_dict(state["pool"])


class SeparabilityTerm(AxiomTerm):
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
        if ctx.vector.size(0) == 0:
            return None
        stage = self._stages.get(ctx.vector.size(-1))
        if stage is None:
            return None

        pooled = stage.pool(ctx.vector.to(stage.pool.score.weight.dtype))
        t_emb = F.normalize(pooled.float(), dim=-1)

        negatives = list(stage.queue)
        stage.queue.append(t_emb.detach())
        if not negatives:
            return None

        neg_sims = torch.stack([torch.dot(t_emb, t_neg) for t_neg in negatives])
        return torch.logsumexp(neg_sims / TEMPERATURE, dim=0)
