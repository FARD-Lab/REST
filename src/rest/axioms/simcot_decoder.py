from __future__ import annotations

import copy
from typing import Any, Dict, List

import torch
import torch.nn as nn


class CloneDecoderStage:
    def __init__(self, source_model: nn.Module, device: torch.device, dtype: torch.dtype):
        self.decoder = copy.deepcopy(source_model).to(device=device, dtype=dtype)
        self.decoder.train()
        if hasattr(self.decoder, "config") and hasattr(self.decoder.config, "use_cache"):
            self.decoder.config.use_cache = False
        for param in self.decoder.parameters():
            param.requires_grad_(True)

    def parameters(self) -> List[torch.nn.Parameter]:
        return list(self.decoder.parameters())

    def state_dict(self) -> Dict[str, Any]:
        return self.decoder.state_dict()

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.decoder.load_state_dict(state)
