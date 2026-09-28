from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass
class HandoffContext:
    downstream_model: torch.nn.Module
    downstream_tokenizer: object
    downstream_embedding_layer: torch.nn.Module
    reference_prompt: str
    upstream_text: str
    downstream_text: str
    vector: torch.Tensor
    trained_logits: torch.Tensor
    trained_assistant_mask: torch.Tensor
    enable_thinking: bool
    device: torch.device
    max_length: int
    embed_dtype: torch.dtype
    upstream_logits: Optional[torch.Tensor] = None
    upstream_assistant_mask: Optional[torch.Tensor] = None
    upstream_input_text: Optional[str] = None
    trained_hidden_states: Optional[Tuple[torch.Tensor, ...]] = None


class AxiomTerm(abc.ABC):
    def __init__(self, weight: float, **kwargs):
        self.weight = weight

    def parameters(self) -> List[torch.nn.Parameter]:
        return []

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        pass

    @abc.abstractmethod
    def compute(self, ctx: HandoffContext) -> Optional[torch.Tensor]:
        raise NotImplementedError
