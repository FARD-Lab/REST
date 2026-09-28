from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
from common import compute_solver_ce_loss, ids_to_embeds, text_to_ids

from .base import AxiomTerm, HandoffContext
from .simcot_decoder import CloneDecoderStage

STAGE_NAMES = ("planner", "refiner", "solver")


def compute_step_reconstruction_loss(
    decoder: nn.Module,
    tokenizer,
    vector: torch.Tensor,
    target_text: str,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if not target_text:
        return None
    if vector.size(0) == 0:
        return None

    target_ids = text_to_ids(tokenizer, target_text)
    if tokenizer.eos_token_id is not None:
        target_ids = target_ids + [int(tokenizer.eos_token_id)]
    if not target_ids:
        return None

    decoder_embed = decoder.get_input_embeddings()
    embed_dtype = decoder_embed.weight.dtype
    target_embeds = ids_to_embeds(decoder_embed, target_ids, device=device, dtype=embed_dtype)
    vector_embeds = vector.to(embed_dtype)

    inputs_embeds = torch.cat([vector_embeds, target_embeds], dim=0).unsqueeze(0)
    attention_mask = torch.ones((inputs_embeds.size(1),), dtype=torch.long, device=device).unsqueeze(0)

    labels = torch.full((inputs_embeds.size(1),), -100, dtype=torch.long, device=device)
    labels[vector.size(0) :] = torch.tensor(target_ids, dtype=torch.long, device=device)

    out = decoder(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    return compute_solver_ce_loss(out.logits, labels.unsqueeze(0))


class SimCotStepTerm(AxiomTerm):
    def __init__(
        self,
        weight: float,
        planner_hidden_size: Optional[int] = None,
        refiner_hidden_size: Optional[int] = None,
        solver_hidden_size: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        planner_model: Optional[nn.Module] = None,
        refiner_model: Optional[nn.Module] = None,
        solver_model: Optional[nn.Module] = None,
        simcot_stages: Optional[Sequence[str]] = None,
        **kwargs,
    ):
        super().__init__(weight)
        stage_sources = {"planner": planner_model, "refiner": refiner_model, "solver": solver_model}
        missing = [name for name, model in stage_sources.items() if model is None]
        if missing:
            raise ValueError(
                "SimCotStepTerm requires the live planner/refiner/solver model objects to clone "
                f"its step decoders from; missing: {missing}. Pass them to build_axiom_term as "
                "planner_model=/refiner_model=/solver_model=."
            )

        selected = tuple(stage_sources) if simcot_stages is None else tuple(simcot_stages)
        unknown = [name for name in selected if name not in stage_sources]
        if unknown:
            raise ValueError(f"Unknown simcot_stages {unknown}. Available: {sorted(stage_sources)}.")
        if not selected:
            raise ValueError("simcot_stages is empty -- SimCotStepTerm would contribute no loss.")

        self._stages: Dict[str, CloneDecoderStage] = {
            name: CloneDecoderStage(stage_sources[name], device, dtype) for name in selected
        }
        self._stage_name_by_model_id: Dict[int, str] = {id(stage_sources[name]): name for name in selected}
        print(f"[simcot] step decoders on stage(s): {list(self._stages)}", flush=True)

    def parameters(self) -> List[torch.nn.Parameter]:
        params: List[torch.nn.Parameter] = []
        for stage in self._stages.values():
            params += stage.parameters()
        return params

    def state_dict(self) -> Dict[str, object]:
        return {name: stage.state_dict() for name, stage in self._stages.items()}

    def load_state_dict(self, state: Dict[str, object]) -> None:
        unexpected = [name for name in state if name not in self._stages]
        if unexpected:
            raise ValueError(
                f"Checkpoint carries SIM-CoT step decoders for stage(s) {unexpected}, but this run "
                f"built decoders for {sorted(self._stages)} only. Resume with the same "
                "--simcot_stages the checkpoint was trained with."
            )
        for name, stage_state in state.items():
            self._stages[name].load_state_dict(stage_state)

    def compute(self, ctx: HandoffContext) -> Optional[torch.Tensor]:
        stage_name = self._stage_name_by_model_id.get(id(ctx.downstream_model))
        if stage_name is None:
            return None
        stage = self._stages[stage_name]
        return compute_step_reconstruction_loss(
            stage.decoder,
            ctx.downstream_tokenizer,
            ctx.vector,
            ctx.upstream_text,
            ctx.device,
        )
