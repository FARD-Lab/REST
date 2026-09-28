from __future__ import annotations

from typing import Optional

import torch
from common import build_stage_with_slot, compute_solver_ce_loss

from .base import AxiomTerm, HandoffContext

NOTE_ONLY_SLOT = "<<MINIMALITY_NOTE_ONLY_SLOT>>"
RESULT_AND_NOTE_SLOT = "<<MINIMALITY_RESULT_AND_NOTE_SLOT>>"


def build_note_only_prompt() -> str:
    return (
        "Here is a compressed note: "
        f"{NOTE_ONLY_SLOT}"
        ". Write out, as precisely as possible, the text this note represents."
    )


def build_result_and_note_prompt(known_result_text: str) -> str:
    return (
        f"The following is a result: {known_result_text}. "
        "Here is a compressed note that, together with an original input you cannot see, "
        "was used to help produce that result: "
        f"{RESULT_AND_NOTE_SLOT}"
        ". Reconstruct, as precisely as possible, the original input."
    )


def compute_ce_y_given_t(
    model: torch.nn.Module,
    tokenizer,
    embedding_layer: torch.nn.Module,
    target_text: str,
    vector: torch.Tensor,
    enable_thinking: bool,
    device: torch.device,
    max_length: int,
    embed_dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    pack = build_stage_with_slot(
        tokenizer=tokenizer,
        embedding_layer=embedding_layer,
        user_prompt_with_slot=build_note_only_prompt(),
        assistant_text=target_text,
        slot_text=NOTE_ONLY_SLOT,
        slot_embeds=vector,
        enable_thinking=enable_thinking,
        device=device,
        embed_dtype=embed_dtype,
        max_length=max_length,
    )
    if not bool(pack.assistant_mask.any()):
        return None

    out = model(
        inputs_embeds=pack.inputs_embeds,
        attention_mask=pack.attention_mask,
        use_cache=False,
        return_dict=True,
    )
    return compute_solver_ce_loss(out.logits, pack.labels)


def compute_ce_x_given_yt(
    model: torch.nn.Module,
    tokenizer,
    embedding_layer: torch.nn.Module,
    known_result_text: str,
    target_text: str,
    vector: torch.Tensor,
    enable_thinking: bool,
    device: torch.device,
    max_length: int,
    embed_dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    pack = build_stage_with_slot(
        tokenizer=tokenizer,
        embedding_layer=embedding_layer,
        user_prompt_with_slot=build_result_and_note_prompt(known_result_text),
        assistant_text=target_text,
        slot_text=RESULT_AND_NOTE_SLOT,
        slot_embeds=vector,
        enable_thinking=enable_thinking,
        device=device,
        embed_dtype=embed_dtype,
        max_length=max_length,
    )
    if not bool(pack.assistant_mask.any()):
        return None

    out = model(
        inputs_embeds=pack.inputs_embeds,
        attention_mask=pack.attention_mask,
        use_cache=False,
        return_dict=True,
    )
    return compute_solver_ce_loss(out.logits, pack.labels)


class MinimalityTerm(AxiomTerm):
    def __init__(self, weight: float, composed: bool = False, **kwargs):
        super().__init__(weight, **kwargs)
        self.composed = bool(composed)

    def compute(self, ctx: HandoffContext) -> Optional[torch.Tensor]:
        if not ctx.upstream_text:
            return None
        if not ctx.upstream_input_text:
            return None

        ce_y_given_t = compute_ce_y_given_t(
            ctx.downstream_model,
            ctx.downstream_tokenizer,
            ctx.downstream_embedding_layer,
            ctx.upstream_text,
            ctx.vector,
            ctx.enable_thinking,
            ctx.device,
            ctx.max_length,
            ctx.embed_dtype,
        )
        if ce_y_given_t is None:
            return None

        if self.composed:
            return ce_y_given_t

        ce_x_given_yt = compute_ce_x_given_yt(
            ctx.downstream_model,
            ctx.downstream_tokenizer,
            ctx.downstream_embedding_layer,
            ctx.upstream_text,
            ctx.upstream_input_text,
            ctx.vector,
            ctx.enable_thinking,
            ctx.device,
            ctx.max_length,
            ctx.embed_dtype,
        )
        if ce_x_given_yt is None:
            return None

        return ce_y_given_t - 0.3 * ce_x_given_yt
