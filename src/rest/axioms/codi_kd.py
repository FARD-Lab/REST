from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .base import AxiomTerm, HandoffContext
from .causality import first_assistant_logit_index
from .codi_reference_forward import run_codi_reference_forward


class CodiKdTerm(AxiomTerm):
    def __init__(self, weight: float, **kwargs):
        super().__init__(weight)
        self._distill_loss_fct = nn.SmoothL1Loss()

    def compute(self, ctx: HandoffContext) -> Optional[torch.Tensor]:
        student_hidden_states = ctx.trained_hidden_states
        if student_hidden_states is None:
            raise RuntimeError("CodiKdTerm requires HandoffContext.trained_hidden_states.")

        student_boundary = first_assistant_logit_index(ctx.trained_assistant_mask)
        if student_boundary is None:
            return None

        reference = run_codi_reference_forward(
            ctx.downstream_model,
            ctx.downstream_tokenizer,
            ctx.reference_prompt,
            ctx.downstream_text,
            ctx.enable_thinking,
            ctx.device,
            ctx.max_length,
        )
        if reference is None or reference.hidden_states is None:
            return None
        if len(reference.hidden_states) != len(student_hidden_states):
            return None

        layer_losses = []
        for teacher_layer, student_layer in zip(reference.hidden_states, student_hidden_states):
            teacher_probe = teacher_layer[:, reference.boundary, :].float().detach()
            student_probe = student_layer[:, student_boundary, :].float()
            layer_losses.append(self._distill_loss_fct(student_probe, teacher_probe))

        return torch.stack(layer_losses).mean()
