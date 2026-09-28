from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from .causality import build_reference_teacher_forced_inputs, first_assistant_logit_index


@dataclass
class CodiReferenceForward:
    boundary: int
    hidden_states: Tuple[torch.Tensor, ...]


def run_codi_reference_forward(
    model: torch.nn.Module,
    tokenizer,
    user_prompt: str,
    assistant_text: str,
    enable_thinking: bool,
    device: torch.device,
    max_length: int,
) -> Optional[CodiReferenceForward]:
    if not user_prompt:
        return None

    input_ids, attention_mask, assistant_mask = build_reference_teacher_forced_inputs(
        tokenizer,
        user_prompt,
        assistant_text,
        enable_thinking,
        device,
        max_length,
    )
    boundary = first_assistant_logit_index(assistant_mask)
    if boundary is None:
        return None

    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

    return CodiReferenceForward(boundary=boundary, hidden_states=output.hidden_states)
