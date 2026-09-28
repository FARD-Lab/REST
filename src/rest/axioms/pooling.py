from __future__ import annotations

import torch
import torch.nn as nn


class AttentionPool(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, steps: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(steps).squeeze(-1), dim=0)
        return (weights.unsqueeze(-1) * steps).sum(dim=0)
