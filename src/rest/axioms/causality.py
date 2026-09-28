from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from common import render_chat_ids

from .base import AxiomTerm, HandoffContext


def build_reference_teacher_forced_inputs(
    tokenizer,
    user_prompt: str,
    assistant_text: str,
    enable_thinking: bool,
    device: torch.device,
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prompt_ids = render_chat_ids(
        tokenizer,
        user_prompt,
        assistant_text=None,
        enable_thinking=enable_thinking,
        max_length=max_length,
    )
    full_ids = render_chat_ids(
        tokenizer,
        user_prompt,
        assistant_text=assistant_text,
        enable_thinking=enable_thinking,
        max_length=max_length,
    )
    assistant_token_count = max(len(full_ids) - len(prompt_ids), 0)
    if len(full_ids) > max_length:
        full_ids = full_ids[-max_length:]
        assistant_token_count = min(assistant_token_count, len(full_ids))
    prompt_len = len(full_ids) - assistant_token_count

    assistant_mask = torch.zeros((len(full_ids),), dtype=torch.bool, device=device)
    if prompt_len < len(full_ids):
        assistant_mask[prompt_len:] = True

    input_ids = torch.tensor(full_ids, dtype=torch.long, device=device).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask, assistant_mask


def first_assistant_logit_index(assistant_mask: torch.Tensor) -> Optional[int]:
    nonzero = assistant_mask.nonzero(as_tuple=True)[0]
    if nonzero.numel() == 0:
        return None
    first_idx = int(nonzero[0].item())
    if first_idx == 0:
        return None
    return first_idx - 1


def causality_kl_loss(reference_logits: torch.Tensor, trained_logits: torch.Tensor) -> torch.Tensor:
    target_probs = torch.softmax(reference_logits.float(), dim=-1)
    trained_log_probs = torch.log_softmax(trained_logits.float(), dim=-1)
    return F.kl_div(trained_log_probs, target_probs, reduction="batchmean")


def compute_causality_term(
    ref_model: torch.nn.Module,
    ref_tokenizer,
    reference_user_prompt: str,
    reference_assistant_text: str,
    trained_logits: torch.Tensor,
    trained_assistant_mask: torch.Tensor,
    enable_thinking: bool,
    device: torch.device,
    max_length: int,
) -> Optional[torch.Tensor]:
    if not reference_user_prompt:
        return None
    trained_boundary = first_assistant_logit_index(trained_assistant_mask)
    if trained_boundary is None:
        return None
    trained_count = int(trained_assistant_mask.sum().item())

    ref_ids, ref_mask, ref_assistant_mask = build_reference_teacher_forced_inputs(
        ref_tokenizer,
        reference_user_prompt,
        reference_assistant_text,
        enable_thinking,
        device,
        max_length,
    )
    ref_boundary = first_assistant_logit_index(ref_assistant_mask)
    if ref_boundary is None:
        return None
    ref_count = int(ref_assistant_mask.sum().item())

    n_positions = min(trained_count, ref_count)
    if n_positions <= 0:
        return None

    with torch.no_grad():
        ref_out = ref_model(input_ids=ref_ids, attention_mask=ref_mask, use_cache=False, return_dict=True)

    trained_slice = trained_logits[
        0, trained_boundary + trained_count - n_positions : trained_boundary + trained_count, :
    ]
    ref_slice = ref_out.logits[0, ref_boundary + ref_count - n_positions : ref_boundary + ref_count, :]
    return causality_kl_loss(ref_slice, trained_slice)


class CausalityTerm(AxiomTerm):
    def compute(self, ctx: HandoffContext) -> Optional[torch.Tensor]:
        return compute_causality_term(
            ctx.downstream_model,
            ctx.downstream_tokenizer,
            ctx.reference_prompt,
            ctx.downstream_text,
            ctx.trained_logits,
            ctx.trained_assistant_mask,
            ctx.enable_thinking,
            ctx.device,
            ctx.max_length,
        )
